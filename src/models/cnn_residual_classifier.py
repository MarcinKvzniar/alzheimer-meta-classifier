"""
Custom CNN classifier for Alzheimer's disease classification (4 classes, severe imbalance).
Uses PyTorch, trains from scratch, integrates with CombinedDataset loader.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.combined_data_loader import load_combined_alzheimer_data

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return T.functional.pad(image, padding, 0, 'edge')

class AlzheimerImageDataset(Dataset):
    def __init__(self, combined_dataset, img_size=96, augment=False):
        self.data = combined_dataset.data if hasattr(combined_dataset, 'data') else combined_dataset
        self.img_size = img_size
        self.augment = augment
        self.transform = T.Compose([
            SquarePad(),                  
            T.Resize((128, 128)),        
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        self.aug_transform = T.Compose([
            SquarePad(),
            T.Resize((128, 128)),
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)), 
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        img = item['image']
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.augment:
            img = self.aug_transform(img)
        else:
            img = self.transform(img)
        label = item['label']
        return img, label

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block: Helps the model focus on important channels."""
    def __init__(self, channel, reduction=8): 
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AlzheimerCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        
        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        def make_layer(in_c, out_c, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                SEBlock(out_c), 
            )

        self.layer1 = make_layer(32, 32, stride=1)
        self.layer2 = make_layer(32, 64, stride=2)
        self.layer3 = make_layer(64, 128, stride=2)
        self.layer4 = make_layer(128, 256, stride=2)
        
        self.shortcut2 = nn.Sequential(nn.Conv2d(32, 64, 1, 2), nn.BatchNorm2d(64))
        self.shortcut3 = nn.Sequential(nn.Conv2d(64, 128, 1, 2), nn.BatchNorm2d(128))
        self.shortcut4 = nn.Sequential(nn.Conv2d(128, 256, 1, 2), nn.BatchNorm2d(256))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.3), 
            nn.Linear(256, num_classes) 
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.initial(x)
        
        x = torch.relu(x + self.layer1(x))
        x = torch.relu(self.shortcut2(x) + self.layer2(x))
        x = torch.relu(self.shortcut3(x) + self.layer3(x))
        x = torch.relu(self.shortcut4(x) + self.layer4(x))

        return self.classifier(x)
    
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

def compute_class_weights(dataset, num_classes=4):
    labels = [item['label'] for item in dataset.data]
    counts = np.bincount(labels, minlength=num_classes)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float32)

def train_epoch(model, loader, criterion, optimizer, device, accumulation_steps=4):
    model.train()

    running_loss, correct, total = 0.0, 0, 0
    for i, (imgs, labels) in enumerate(tqdm(loader, desc='Train', leave=False)):
        imgs, labels = imgs.to(device), labels.to(device)
        
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        
        loss = loss / accumulation_steps 
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
        
        running_loss += (loss.item() * accumulation_steps) * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    if (i + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
        
    return running_loss / total, correct / total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Eval', leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


# GLOBAL PARAMS
EPOCHS = 60
BATCH_SIZE = 16
ACCUMULATION_STEPS = 4

IMG_SIZE = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.02
SAVE_PATH = 'models/cnn_model.pt'
USE_CUDA = torch.cuda.is_available()

def main():
    device = torch.device('cuda' if USE_CUDA else 'cpu')
    print(f"Using device: {device}")

    print("\nLoading combined dataset")
    train_data, val_data, test_data = load_combined_alzheimer_data(
        use_huggingface=True,
        use_local=True,
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )

    class_weights = compute_class_weights(train_data)
    print(f"Class weights: {class_weights}")

    train_ds = AlzheimerImageDataset(train_data, img_size=IMG_SIZE, augment=True)
    val_ds = AlzheimerImageDataset(val_data, img_size=IMG_SIZE, augment=False)
    test_ds = AlzheimerImageDataset(test_data, img_size=IMG_SIZE, augment=False)

    train_labels = [item['label'] for item in train_data]
    
    model = AlzheimerCNN(num_classes=4).to(device)
    
    class_sample_counts = np.bincount(train_labels)
    weight = 1. / class_sample_counts
    weight = np.power(weight, 0.5)
    samples_weights = torch.from_numpy(np.array([weight[t] for t in train_labels])).double()
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        sampler=sampler, 
        num_workers=0 if sys.platform == 'win32' else 2,
        pin_memory=False
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    best_val_f1 = 0.0  
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps=ACCUMULATION_STEPS)
        val_loss, val_acc, val_preds, val_labels = eval_epoch(model, val_loader, criterion, device)

        val_f1 = f1_score(val_labels, val_preds, average='macro')
        val_f1_per_class = f1_score(val_labels, val_preds, average=None)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"LR: {current_lr:.6f}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        f1_str = [f"{score:.4f}" for score in val_f1_per_class]
        print(f"Val Class F1s: {f1_str}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"SUCCESS: Model saved to {SAVE_PATH} (Best F1: {best_val_f1:.4f})")
        
        scheduler.step(val_f1)
        
        early_stopping(val_f1)
        if early_stopping.early_stop:
            print("Early stopping triggered. Finished.")
            break

    print("\n Model Evaluation:")
    model.load_state_dict(torch.load(SAVE_PATH))
    test_loss, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, digits=4))
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, test_preds))

if __name__ == '__main__':
    main()
