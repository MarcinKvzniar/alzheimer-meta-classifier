"""
Combined data loader that merges HuggingFace and local datasets.
Provides unified interface for loading both data sources together.
"""

import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
from sklearn.model_selection import train_test_split
import random

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import AlzheimerDataLoader
from src.data.local_data_loader import LocalAlzheimerDataLoader


class CombinedDataset:
    """
    Wrapper class to make combined data compatible with classifier interfaces.
    Mimics HuggingFace Dataset behavior for seamless integration.
    """
    
    def __init__(self, data_list: List[Dict[str, Any]]):
        """
        Initialize with a list of data dictionaries.
        
        Args:
            data_list: List of dicts with 'image' and 'label' keys
        """
        self.data = data_list
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if isinstance(idx, int):
            return self.data[idx]
        elif isinstance(idx, slice):
            return CombinedDataset(self.data[idx])
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")
    
    def __iter__(self):
        return iter(self.data)
    
    def select(self, indices: List[int]) -> 'CombinedDataset':
        """
        Select subset of data by indices (mimics HuggingFace Dataset.select).
        
        Args:
            indices: List of indices to select
            
        Returns:
            New CombinedDataset with selected items
        """
        selected_data = [self.data[i] for i in indices]
        return CombinedDataset(selected_data)


class CombinedAlzheimerDataLoader:
    """
    Combined data loader that merges HuggingFace and local datasets.
    Provides unified train/val/test splits from both sources.
    """
    
    def __init__(
        self, 
        use_huggingface: bool = True,
        use_local: bool = True,
        local_data_dir: str = None,
        hf_dataset_name: str = 'Falah/Alzheimer_MRI'
    ):
        """
        Initialize the combined data loader.
        
        Args:
            use_huggingface: Whether to include HuggingFace dataset
            use_local: Whether to include local dataset
            local_data_dir: Path to local dataset directory
            hf_dataset_name: Name of HuggingFace dataset
        """
        self.use_huggingface = use_huggingface
        self.use_local = use_local
        self.local_data_dir = local_data_dir
        self.hf_dataset_name = hf_dataset_name
        
        self.hf_loader = None
        self.local_loader = None
        self.combined_data = []
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        self.class_names = {
            0: 'Mild Demented',
            1: 'Moderate Demented',
            2: 'Non Demented',
            3: 'Very Mild Demented'
        }
    
    def load_data(self) -> List[Dict[str, Any]]:
        """
        Load and combine data from both sources.
        
        Returns:
            Combined list of data dictionaries
        """
        print("=" * 80)
        print("COMBINED DATA LOADER")
        print("=" * 80)
        
        all_data = []
        
        if self.use_huggingface:
            print("\n[1/2] Loading HuggingFace Dataset...")
            print("-" * 80)
            try:
                self.hf_loader = AlzheimerDataLoader(self.hf_dataset_name)
                hf_dataset = self.hf_loader.load_data(split='train')
                
                hf_data = []
                for i in range(len(hf_dataset)):
                    hf_data.append({
                        'image': hf_dataset[i]['image'],
                        'label': hf_dataset[i]['label'],
                        'source': 'huggingface'
                    })
                
                all_data.extend(hf_data)
                print(f"✓ HuggingFace data loaded: {len(hf_data)} images")
            except Exception as e:
                print(f"✗ Error loading HuggingFace dataset: {e}")
                if not self.use_local:
                    raise
        
        if self.use_local:
            print("\n[2/2] Loading Local Dataset...")
            print("-" * 80)
            try:
                self.local_loader = LocalAlzheimerDataLoader(self.local_data_dir)
                self.local_loader.load_data()
                
                local_data = []
                for img_path, label in zip(self.local_loader.images, self.local_loader.labels):
                    local_data.append({
                        'image': self.local_loader.load_image(img_path),
                        'label': label,
                        'source': 'local',
                        'path': str(img_path)
                    })
                
                all_data.extend(local_data)
                print(f"✓ Local data loaded: {len(local_data)} images")
            except Exception as e:
                print(f"✗ Error loading local dataset: {e}")
                if not self.use_huggingface:
                    raise
        
        if not all_data:
            raise ValueError("No data loaded from any source!")
        
        self.combined_data = all_data
        
        print("\n" + "=" * 80)
        print(f"TOTAL COMBINED DATA: {len(all_data)} images")
        print("=" * 80)
        
        return all_data
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about the combined dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if not self.combined_data:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        labels = [item['label'] for item in self.combined_data]
        sources = [item['source'] for item in self.combined_data]
        
        labels_array = np.array(labels)
        unique_labels, counts = np.unique(labels_array, return_counts=True)
        
        class_distribution = {
            self.class_names[label]: count 
            for label, count in zip(unique_labels, counts)
        }
        
        source_distribution = {}
        for source in set(sources):
            source_distribution[source] = sources.count(source)
        
        info = {
            'num_examples': len(self.combined_data),
            'num_classes': len(self.class_names),
            'class_distribution': class_distribution,
            'source_distribution': source_distribution,
            'sources': list(source_distribution.keys())
        }
        
        return info
    
    def split_data(
        self, 
        test_size: float = 0.2, 
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True,
        shuffle: bool = True
    ) -> Tuple[CombinedDataset, CombinedDataset, CombinedDataset]:
        """
        Split combined data into train, validation, and test sets.
        
        Args:
            test_size: Proportion for test set
            val_size: Proportion for validation set
            random_state: Random seed for reproducibility
            stratify: Whether to stratify splits by class
            shuffle: Whether to shuffle data before splitting
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        if not self.combined_data:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        print("\n" + "=" * 80)
        print("SPLITTING COMBINED DATA")
        print("=" * 80)
        
        random.seed(random_state)
        np.random.seed(random_state)
        
        data = self.combined_data.copy()
        if shuffle:
            random.shuffle(data)
        
        labels = [item['label'] for item in data]
        indices = np.arange(len(data))
        
        stratify_labels = labels if stratify else None
        
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels
        )
        
        if stratify:
            stratify_train = [labels[i] for i in train_val_idx]
        else:
            stratify_train = None
        
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size / (1 - test_size),
            random_state=random_state,
            stratify=stratify_train
        )
        
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]
        test_data = [data[i] for i in test_idx]
        
        self.train_data = CombinedDataset(train_data)
        self.val_data = CombinedDataset(val_data)
        self.test_data = CombinedDataset(test_data)
        
        print(f"\nData split completed:")
        print(f"  - Training set:   {len(train_data):5d} images")
        print(f"  - Validation set: {len(val_data):5d} images")
        print(f"  - Test set:       {len(test_data):5d} images")
        
        print("\nSource distribution per split:")
        for split_name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
            sources = [item['source'] for item in split_data]
            hf_count = sources.count('huggingface') if 'huggingface' in sources else 0
            local_count = sources.count('local') if 'local' in sources else 0
            print(f"  {split_name:5s}: HuggingFace={hf_count:4d}, Local={local_count:4d}")
        
        print("\nClass distribution per split:")
        for split_name, split_data in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
            split_labels = [item['label'] for item in split_data]
            unique, counts = np.unique(split_labels, return_counts=True)
            print(f"  {split_name:5s}:", end='')
            for label, count in zip(unique, counts):
                class_name = self.class_names[label]
                print(f" {class_name}={count}", end='')
            print()
        
        print("=" * 80)
        
        return self.train_data, self.val_data, self.test_data
    
    def print_sample(self, num_samples: int = 5):
        """
        Print sample examples from the combined dataset.
        
        Args:
            num_samples: Number of samples to print
        """
        if not self.combined_data:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        print(f"\nSample data ({num_samples} examples):")
        print("-" * 80)
        
        indices = random.sample(range(len(self.combined_data)), min(num_samples, len(self.combined_data)))
        
        for i, idx in enumerate(indices, 1):
            item = self.combined_data[idx]
            label = item['label']
            class_name = self.class_names[label]
            source = item['source']
            
            print(f"Example {i}:")
            print(f"  Source: {source}")
            print(f"  Label: {label} - {class_name}")
            if 'path' in item:
                print(f"  Path: {item['path']}")
            try:
                print(f"  Image size: {item['image'].size}")
            except:
                print(f"  Image: <PIL.Image object>")
            print("-" * 80)


def load_combined_alzheimer_data(
    use_huggingface: bool = True,
    use_local: bool = True,
    local_data_dir: str = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    shuffle: bool = True
) -> Tuple[CombinedDataset, CombinedDataset, CombinedDataset]:
    """
    Convenience function to load and split combined Alzheimer's MRI dataset.
    
    Args:
        use_huggingface: Whether to include HuggingFace dataset
        use_local: Whether to include local dataset
        local_data_dir: Path to local dataset directory
        test_size: Proportion for test set
        val_size: Proportion for validation set
        random_state: Random seed
        shuffle: Whether to shuffle before splitting
    
    Returns:
        Tuple of (train_data, val_data, test_data) as CombinedDataset objects
    """
    loader = CombinedAlzheimerDataLoader(
        use_huggingface=use_huggingface,
        use_local=use_local,
        local_data_dir=local_data_dir
    )
    
    loader.load_data()
    
    train_data, val_data, test_data = loader.split_data(
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        shuffle=shuffle
    )
    
    return train_data, val_data, test_data


def main():
    """
    Test the combined data loader.
    """
    print("\n" + "=" * 80)
    print("COMBINED ALZHEIMER'S DATASET LOADER TEST")
    print("=" * 80)
    
    loader = CombinedAlzheimerDataLoader(
        use_huggingface=True,
        use_local=True
    )
    
    combined_data = loader.load_data()
    
    info = loader.get_dataset_info()
    print("\nCombined Dataset Information:")
    print(f"  Total examples: {info['num_examples']}")
    print(f"  Number of classes: {info['num_classes']}")
    print(f"\n  Source distribution:")
    for source, count in info['source_distribution'].items():
        percentage = (count / info['num_examples']) * 100
        print(f"    {source:15s}: {count:5d} ({percentage:5.2f}%)")
    print(f"\n  Class distribution:")
    for class_name, count in info['class_distribution'].items():
        percentage = (count / info['num_examples']) * 100
        print(f"    {class_name:20s}: {count:5d} ({percentage:5.2f}%)")
    
    loader.print_sample(num_samples=3)
    
    train_data, val_data, test_data = loader.split_data(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    print("\n" + "=" * 80)
    print("COMBINED DATA LOADER TEST COMPLETED!")
    print("=" * 80)
    print(f"\nYou now have {len(combined_data)} total images for training!")
    print("This combines both HuggingFace and local datasets for better model performance.")
    print("=" * 80)


if __name__ == '__main__':
    main()
