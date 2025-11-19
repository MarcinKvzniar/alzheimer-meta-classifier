"""
Local dataset loader for Alzheimer's MRI images from local filesystem.
Loads images from folder structure where each subfolder represents a class.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class LocalAlzheimerDataLoader:
    """
    Data loader for local Alzheimer's MRI dataset.
    Expects folder structure:
        datasets/
            MildDemented/
            ModerateDemented/
            NonDemented/
            VeryMildDemented/
    
    Attributes:
        data_dir (Path): Root directory containing class folders
        class_mapping (dict): Mapping from folder names to numeric labels
        images (list): List of image paths
        labels (list): List of corresponding labels
        train_images, val_images, test_images: Split image paths
        train_labels, val_labels, test_labels: Split labels
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the local data loader.
        
        Args:
            data_dir: Path to datasets directory (defaults to project_root/datasets)
        """
        if data_dir is None:
            self.data_dir = project_root / "datasets"
        else:
            self.data_dir = Path(data_dir)
        
        # Class mapping to match HuggingFace dataset
        self.class_mapping = {
            'MildDemented': 0,
            'ModerateDemented': 1,
            'NonDemented': 2,
            'VeryMildDemented': 3
        }
        
        self.class_names = {
            0: 'Mild Demented',
            1: 'Moderate Demented',
            2: 'Non Demented',
            3: 'Very Mild Demented'
        }
        
        self.images = []
        self.labels = []
        self.train_images = None
        self.val_images = None
        self.test_images = None
        self.train_labels = None
        self.val_labels = None
        self.test_labels = None
    
    def load_data(self) -> Tuple[List[Path], List[int]]:
        """
        Load all images and labels from the dataset directory.
        
        Returns:
            Tuple of (image_paths, labels)
        """
        print(f"Loading local dataset from '{self.data_dir}'...")
        
        if not self.data_dir.exists():
            raise ValueError(f"Dataset directory not found: {self.data_dir}")
        
        self.images = []
        self.labels = []
        
        # Supported image formats
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Load images from each class folder
        for class_name, label in self.class_mapping.items():
            class_dir = self.data_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            # Get all image files in the directory
            image_files = [
                f for f in class_dir.iterdir() 
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            
            print(f"  {class_name}: {len(image_files)} images")
            
            for img_path in image_files:
                self.images.append(img_path)
                self.labels.append(label)
        
        print(f"\nDataset loaded successfully. Total images: {len(self.images)}")
        
        return self.images, self.labels
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.images:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        labels_array = np.array(self.labels)
        unique_labels, counts = np.unique(labels_array, return_counts=True)
        class_distribution = {
            self.class_names[label]: count 
            for label, count in zip(unique_labels, counts)
        }
        
        info = {
            'num_examples': len(self.images),
            'num_classes': len(self.class_mapping),
            'class_mapping': self.class_mapping,
            'class_names': self.class_names,
            'class_distribution': class_distribution,
            'data_dir': str(self.data_dir)
        }
        
        return info
    
    def split_data(
        self, 
        test_size: float = 0.2, 
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[Tuple[List, List], Tuple[List, List], Tuple[List, List]]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining data)
            random_state: Random seed for reproducibility
            stratify: Whether to stratify splits by class
            
        Returns:
            Tuple of ((train_images, train_labels), (val_images, val_labels), (test_images, test_labels))
        """
        if not self.images:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        stratify_labels = self.labels if stratify else None
        
        # First split: separate test set
        train_val_images, test_images, train_val_labels, test_labels = train_test_split(
            self.images,
            self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels
        )
        
        # Second split: separate validation from training
        stratify_train = train_val_labels if stratify else None
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_val_images,
            train_val_labels,
            test_size=val_size / (1 - test_size),
            random_state=random_state,
            stratify=stratify_train
        )
        
        self.train_images = train_images
        self.val_images = val_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels
        
        print(f"\nData split completed:")
        print(f"  - Training set: {len(train_images)} images")
        print(f"  - Validation set: {len(val_images)} images")
        print(f"  - Test set: {len(test_images)} images")
        
        return (
            (train_images, train_labels),
            (val_images, val_labels),
            (test_images, test_labels)
        )
    
    def load_image(self, image_path: Path) -> Image.Image:
        """
        Load a single image from disk.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image
        """
        return Image.open(image_path).convert('RGB')
    
    def get_batch(
        self, 
        image_paths: List[Path], 
        labels: List[int],
        indices: Optional[List[int]] = None
    ) -> Tuple[List[Image.Image], List[int]]:
        """
        Get a batch of images and labels.
        
        Args:
            image_paths: List of image paths
            labels: List of labels
            indices: Optional list of indices to select from
            
        Returns:
            Tuple of (images, labels)
        """
        if indices is None:
            indices = range(len(image_paths))
        
        batch_images = []
        batch_labels = []
        
        for idx in indices:
            img = self.load_image(image_paths[idx])
            batch_images.append(img)
            batch_labels.append(labels[idx])
        
        return batch_images, batch_labels
    
    def create_dataset_dict(self, split: str = 'all') -> List[Dict[str, Any]]:
        """
        Create a dataset dictionary compatible with HuggingFace datasets format.
        
        Args:
            split: Which split to create ('train', 'val', 'test', or 'all')
            
        Returns:
            List of dictionaries with 'image' and 'label' keys
        """
        if split == 'train':
            if self.train_images is None:
                raise ValueError("Data not split. Call split_data() first.")
            image_paths = self.train_images
            labels = self.train_labels
        elif split == 'val':
            if self.val_images is None:
                raise ValueError("Data not split. Call split_data() first.")
            image_paths = self.val_images
            labels = self.val_labels
        elif split == 'test':
            if self.test_images is None:
                raise ValueError("Data not split. Call split_data() first.")
            image_paths = self.test_images
            labels = self.test_labels
        else:  # 'all'
            if not self.images:
                raise ValueError("Dataset not loaded. Call load_data() first.")
            image_paths = self.images
            labels = self.labels
        
        dataset = []
        print(f"\nCreating dataset dictionary for '{split}' split...")
        for i, (img_path, label) in enumerate(zip(image_paths, labels)):
            if i % 500 == 0 and i > 0:
                print(f"  Processed {i}/{len(image_paths)} images...")
            
            dataset.append({
                'image': self.load_image(img_path),
                'label': label,
                'path': str(img_path)
            })
        
        print(f"Dataset dictionary created with {len(dataset)} examples")
        return dataset
    
    def print_sample(self, num_samples: int = 5):
        """
        Print sample examples from the dataset.
        
        Args:
            num_samples: Number of samples to print
        """
        if not self.images:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        print(f"\nSample data ({num_samples} examples):")
        print("-" * 80)
        
        indices = np.random.choice(len(self.images), min(num_samples, len(self.images)), replace=False)
        
        for i, idx in enumerate(indices, 1):
            img_path = self.images[idx]
            label = self.labels[idx]
            class_name = self.class_names[label]
            
            print(f"Example {i}:")
            print(f"  Path: {img_path}")
            print(f"  Label: {label} - {class_name}")
            
            # Try to get image size
            try:
                img = self.load_image(img_path)
                print(f"  Image size: {img.size}")
            except Exception as e:
                print(f"  Error loading image: {e}")
            
            print("-" * 80)


def load_local_alzheimer_data(
    data_dir: str = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Convenience function to load and split local Alzheimer's MRI dataset.
    Returns data in HuggingFace-compatible format.
    
    Args:
        data_dir: Path to datasets directory
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
    
    Returns:
        Tuple of (train_data, val_data, test_data) as lists of dictionaries
    """
    loader = LocalAlzheimerDataLoader(data_dir)
    loader.load_data()
    loader.split_data(
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
    
    train_data = loader.create_dataset_dict('train')
    val_data = loader.create_dataset_dict('val')
    test_data = loader.create_dataset_dict('test')
    
    return train_data, val_data, test_data


def main():
    """
    Test the local data loader.
    """
    print("=" * 80)
    print("LOCAL ALZHEIMER'S DATASET LOADER TEST")
    print("=" * 80)
    
    # Initialize loader
    loader = LocalAlzheimerDataLoader()
    
    # Load data
    images, labels = loader.load_data()
    
    # Get dataset info
    info = loader.get_dataset_info()
    print("\nDataset Information:")
    print(f"  Number of examples: {info['num_examples']}")
    print(f"  Number of classes: {info['num_classes']}")
    print(f"  Data directory: {info['data_dir']}")
    print(f"\n  Class distribution:")
    for class_name, count in info['class_distribution'].items():
        percentage = (count / info['num_examples']) * 100
        print(f"    {class_name:20s}: {count:5d} ({percentage:5.2f}%)")
    
    # Print samples
    loader.print_sample(num_samples=3)
    
    # Split data
    (train_imgs, train_lbls), (val_imgs, val_lbls), (test_imgs, test_lbls) = loader.split_data(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    print("\n" + "=" * 80)
    print("LOCAL DATA LOADER TEST COMPLETED!")
    print("=" * 80)


if __name__ == '__main__':
    main()
