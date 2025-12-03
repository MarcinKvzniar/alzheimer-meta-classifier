"""
Meta Predictor module for ensemble predictions using multiple trained classifiers.
Combines Random Forest, SVM, Gradient Boosting and CNNs for robust predictions.
"""

import sys
import os
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Any
import random
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.random_forest_classifier import RandomForestAlzheimerClassifier
from src.models.svm_classifier import SVMAlzheimerClassifier
from src.models.gradient_boosting_classifier import GradientBoostingAlzheimerClassifier
from src.data.combined_data_loader import load_combined_alzheimer_data

class SEBlock(nn.Module):
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
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class AlzheimerResidualCNN(nn.Module):
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

    def forward(self, x):
        x = self.initial(x)
        x = torch.relu(x + self.layer1(x))
        x = torch.relu(self.shortcut2(x) + self.layer2(x))
        x = torch.relu(self.shortcut3(x) + self.layer3(x))
        x = torch.relu(self.shortcut4(x) + self.layer4(x))
        return self.classifier(x)


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.conv2(torch.relu(self.bn2(out)))
        return torch.cat([x, out], 1)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x):
        out = self.conv(torch.relu(self.bn(x)))
        out = self.pool(out)
        return out


class AlzheimerDenseNet(nn.Module):
    def __init__(self, num_classes=4, growth_rate=32, reduction=0.5):
        super().__init__()
        num_init_features = 2 * growth_rate
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        num_features = num_init_features
        block_config = (6, 12, 24)

        for i, num_layers in enumerate(block_config):
            block = self._make_dense_block(num_features, num_layers, growth_rate)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                out_features = int(num_features * reduction)
                trans = TransitionLayer(num_features, out_features)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = out_features

        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, num_classes)
        )

    def _make_dense_block(self, in_channels, num_layers, growth_rate):
        layers = []
        for i in range(num_layers):
            layer = DenseLayer(in_channels + i * growth_rate, growth_rate)
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.features(x)
        out = torch.relu(self.final_bn(features))
        out = self.classifier(out)
        return out


class CNNClassifierWrapper:
    """Wrapper to make CNN models compatible with the meta predictor interface."""
    
    def __init__(self, model, transform, device='cpu'):
        self.model = model
        self.transform = transform
        self.device = device
        self.model.to(device)
        self.model.eval()
        
    def predict(self, image: Image.Image) -> int:
        with torch.no_grad():
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            outputs = self.model(img_tensor)
            _, pred = torch.max(outputs, 1)
            return pred.item()
    
    def predict_proba(self, image: Image.Image) -> np.ndarray:
        with torch.no_grad():
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            outputs = self.model(img_tensor)
            proba = torch.softmax(outputs, dim=1)
            return proba.cpu().numpy()[0]


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return T.functional.pad(image, padding, 0, 'edge')


class MetaPredictor:
    """
    Meta predictor that combines predictions from multiple classifiers.
    Uses ensemble voting and provides detailed analysis.
    """
    
    
    def __init__(self, models_dir: str = None, use_classifiers: List[str] = None):
        if models_dir is None:
            models_dir = project_root / "models"
        else:
            models_dir = Path(models_dir)
        
        self.models_dir = models_dir
        self.classifiers = {}
        self.use_classifiers = use_classifiers or [
            'random_forest', 'svm', 'gradient_boosting', 
            'cnn_basic', 'cnn_residual', 'cnn_dense'
        ]
        self.class_names = {
            0: "Mild Demented",
            1: "Moderate Demented", 
            2: "Non Demented",
            3: "Very Mild Demented"
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.cnn_transform = T.Compose([
            SquarePad(),
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        
    def load_classifiers(self):
        print("=" * 80)
        print("Loading Trained Classifiers")
        print("=" * 80)
        
        if 'random_forest' in self.use_classifiers:
            rf_path = self.models_dir / "random_forest_model.pkl"
            if rf_path.exists():
                print(f"Loading Random Forest classifier...")
                rf_classifier = RandomForestAlzheimerClassifier()
                rf_classifier.load_model(str(rf_path))
                self.classifiers['Random Forest'] = rf_classifier
                print("✓ Random Forest loaded")
        
        if 'svm' in self.use_classifiers:
            svm_path = self.models_dir / "svm_model.pkl"
            if svm_path.exists():
                print(f"Loading SVM classifier...")
                svm_classifier = SVMAlzheimerClassifier()
                svm_classifier.load_model(str(svm_path))
                self.classifiers['SVM'] = svm_classifier
                print("✓ SVM loaded")
        
        if 'gradient_boosting' in self.use_classifiers:
            gb_path = self.models_dir / "gradient_boosting_model.pkl"
            if gb_path.exists():
                print(f"Loading Gradient Boosting classifier...")
                gb_classifier = GradientBoostingAlzheimerClassifier()
                gb_classifier.load_model(str(gb_path))
                self.classifiers['Gradient Boosting'] = gb_classifier
                print("✓ Gradient Boosting loaded")
        
        cnn_configs = [
            ('cnn_residual', 'cnn_residual_model.pt', AlzheimerResidualCNN, 'CNN Residual'),
            ('cnn_dense', 'cnn_dense_model.pt', AlzheimerDenseNet, 'CNN Dense')
        ]

        for conf_key, filename, model_class, display_name in cnn_configs:
            if conf_key in self.use_classifiers:
                path = self.models_dir / filename
                if path.exists():
                    print(f"Loading {display_name}...")
                    try:
                        model = model_class(num_classes=4)
                        model.load_state_dict(torch.load(path, map_location=self.device), strict=False)
                        cnn_wrapper = CNNClassifierWrapper(model, self.cnn_transform, self.device)
                        self.classifiers[display_name] = cnn_wrapper
                        print(f"✓ {display_name} loaded")
                    except Exception as e:
                        print(f"✗ Error loading {display_name}: {e}")

        if not self.classifiers:
            print("WARNING: No classifiers were loaded!")
        else:
            print(f"\nSuccessfully loaded {len(self.classifiers)} classifiers.")
            print("=" * 80)
    
    def predict_single_image(self, image: Image.Image, true_label: int = None) -> Dict[str, Any]:
        """Predict class for a single image using all classifiers."""
        if not self.classifiers:
            raise ValueError("No classifiers loaded.")
        
        results = {
            'individual_predictions': {},
            'individual_probabilities': {},
            'ensemble_prediction': None,
            'ensemble_probability': None,
            'agreement_score': 0.0,
            'true_label': true_label
        }
        
        predictions = []
        all_probabilities = []
        
        for name, classifier in self.classifiers.items():
            pred = classifier.predict(image)
            proba = classifier.predict_proba(image)
            
            results['individual_predictions'][name] = pred
            results['individual_probabilities'][name] = proba
            
            predictions.append(pred)
            all_probabilities.append(proba)
        
        predictions_array = np.array(predictions)
        unique, counts = np.unique(predictions_array, return_counts=True)
        ensemble_pred = unique[np.argmax(counts)]
        
        agreement_count = np.sum(predictions_array == ensemble_pred)
        results['agreement_score'] = agreement_count / len(predictions)
        
        mean_probabilities = np.mean(all_probabilities, axis=0)
        results['ensemble_probability'] = mean_probabilities
        results['ensemble_prediction'] = ensemble_pred
        
        return results
    
    def display_prediction_results(self, results: Dict[str, Any], title: str = "PREDICTION RESULTS"):
        print("\n" + "=" * 80)
        print(f"{title}")
        print("=" * 80)
        
        if results['true_label'] is not None:
            true_label = results['true_label']
            print(f"\n TRUE LABEL: {true_label} - {self.class_names.get(true_label, 'Unknown')}")
            print("-" * 80)
        
        print("\n INDIVIDUAL CLASSIFIER PREDICTIONS:")
        
        for name, pred in results['individual_predictions'].items():
            proba = results['individual_probabilities'][name]
            confidence = proba[pred]
            class_name = self.class_names.get(pred, 'Unknown')
            
            correct_marker = ""
            if results['true_label'] is not None:
                correct_marker = " ✓" if pred == results['true_label'] else " ✗"
            
            print(f"{name:20s}: {pred} - {class_name:<20} (Conf: {confidence:.2%}){correct_marker}")
        
        print("\n" + "-" * 80)
        print(" ENSEMBLE PREDICTION:")
        print("-" * 80)
        
        ensemble_pred = results['ensemble_prediction']
        ensemble_class = self.class_names.get(ensemble_pred, 'Unknown')
        agreement = results['agreement_score']
        
        correct_marker = ""
        if results['true_label'] is not None:
            correct_marker = " ✓" if ensemble_pred == results['true_label'] else " ✗"
        
        print(f"Final Prediction:   {ensemble_pred} - {ensemble_class}{correct_marker}")
        print(f"Ensemble Conf:      {results['ensemble_probability'][ensemble_pred]:.2%}")
        print(f"Agreement Score:    {agreement:.2%} ({int(agreement * len(self.classifiers))}/{len(self.classifiers)} agree)")
        
        print(f"\nMean Probabilities:")
        for label, prob in enumerate(results['ensemble_probability']):
            class_name = self.class_names.get(label, f'Class {label}')
            bar = "█" * int(prob * 20)
            print(f"  {label} - {class_name:20s}: {bar} {prob:.2%}")
        print("=" * 80)

    def evaluate_random_samples(self, dataset, num_samples: int = 5):
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
        print(f"\nEvaluating {len(indices)} random samples...")
        
        for idx in indices:
            example = dataset[idx]
            image = example['image']
            
            if not isinstance(image, Image.Image):
                image = Image.fromarray(np.array(image))
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            true_label = example['label']
            
            results = self.predict_single_image(image, true_label)
            self.display_prediction_results(results, title=f"SAMPLE #{idx}")

    def predict_from_file(self, image_path: str):
        """Load an image from path and run prediction."""
        clean_path = image_path.strip('"').strip("'")
        path_obj = Path(clean_path)
        
        if not path_obj.exists():
            print(f"\n ERROR: File does not exist: {clean_path}")
            return
            
        try:
            print(f"\n Loading image: {path_obj.name}")
            image = Image.open(path_obj).convert('RGB')
            
            results = self.predict_single_image(image, true_label=None)
            self.display_prediction_results(results, title=f"FILE: {path_obj.name}")
            
        except Exception as e:
            print(f"\n ERROR processing image: {e}")


def main():
    print("\n" + "=" * 80)
    print("META-PREDICTOR: ENSEMBLE ALZHEIMER'S DISEASE CLASSIFICATION")
    print("=" * 80)
    
    meta_predictor = MetaPredictor()
    meta_predictor.load_classifiers()
    
    if not meta_predictor.classifiers:
        print("Failed to load any classifiers. Please check the 'models' directory.")
        return

    while True:
        print("\n" + "-" * 40)
        print("SELECT MODE:")
        print("-" * 40)
        print("1. Evaluate random samples from Dataset (Test Set)")
        print("2. Analyze custom image (file path)")
        print("q. Quit")
        
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == '1':
            try:
                print("\nLoading dataset (this may take a moment)...")
                _, _, test_data = load_combined_alzheimer_data(
                    test_size=0.2, val_size=0.1, random_state=42
                )
                
                num = input("How many samples? (default 5): ")
                num = int(num) if num.isdigit() else 5
                
                meta_predictor.evaluate_random_samples(test_data, num_samples=num)
                
            except Exception as e:
                print(f"Error loading data: {e}")
                
        elif choice == '2':
            path = input("\nEnter image file path (jpg/png): ")
            if path:
                meta_predictor.predict_from_file(path)
                
        elif choice == 'q':
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid selection.")

if __name__ == '__main__':
    main()