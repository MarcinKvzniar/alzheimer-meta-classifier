from datasets import load_dataset, Dataset
from typing import Optional, Tuple, Dict, Any
import numpy as np
from sklearn.model_selection import train_test_split


class AlzheimerDataLoader:
    """
    Data loader for Alzheimer's MRI dataset from Hugging Face.
    
    Attributes:
        dataset_name (str): Name of the dataset on Hugging Face Hub
        dataset (Dataset): Loaded dataset
        train_data (Dataset): Training split
        val_data (Dataset): Validation split
        test_data (Dataset): Test split
    """
    
    def __init__(self, dataset_name: str = 'Falah/Alzheimer_MRI'):
        self.dataset_name = dataset_name
        self.dataset = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
    
    def load_data(self, split: str = 'train') -> Dataset:
        print(f"Loading dataset '{self.dataset_name}' (split: {split})...")
        self.dataset = load_dataset(self.dataset_name, split=split)
        print(f"Dataset loaded successfully. Number of examples: {len(self.dataset)}")
        return self.dataset
    
    def get_dataset_info(self) -> Dict[str, Any]:
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        info = {
            'num_examples': len(self.dataset),
            'features': self.dataset.features,
            'column_names': self.dataset.column_names
        }
        
        if 'label' in self.dataset.column_names:
            labels = self.dataset['label']
            unique_labels, counts = np.unique(labels, return_counts=True)
            info['class_distribution'] = dict(zip(unique_labels, counts))
        
        return info
    
    def split_data(
        self, 
        test_size: float = 0.2, 
        val_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[Dataset, Dataset, Dataset]:
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        indices = np.arange(len(self.dataset))
        stratify_labels = self.dataset['label'] if stratify and 'label' in self.dataset.column_names else None
        
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_labels
        )
        
        if stratify_labels is not None:
            stratify_labels_train = [stratify_labels[i] for i in train_val_idx]
        else:
            stratify_labels_train = None
        
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size / (1 - test_size),
            random_state=random_state,
            stratify=stratify_labels_train
        )
        
        self.train_data = self.dataset.select(train_idx)
        self.val_data = self.dataset.select(val_idx)
        self.test_data = self.dataset.select(test_idx)
        
        print(f"Data split completed:")
        print(f"  - Training set: {len(self.train_data)} examples")
        print(f"  - Validation set: {len(self.val_data)} examples")
        print(f"  - Test set: {len(self.test_data)} examples")
        
        return self.train_data, self.val_data, self.test_data
    
    def get_train_data(self) -> Dataset:
        """Get training dataset."""
        if self.train_data is None:
            raise ValueError("Data not split. Call split_data() first.")
        return self.train_data
    
    def get_val_data(self) -> Dataset:
        """Get validation dataset."""
        if self.val_data is None:
            raise ValueError("Data not split. Call split_data() first.")
        return self.val_data
    
    def get_test_data(self) -> Dataset:
        """Get test dataset."""
        if self.test_data is None:
            raise ValueError("Data not split. Call split_data() first.")
        return self.test_data
    
    def print_sample(self, num_samples: int = 5):
        """
        Print sample examples from the dataset.
        
        Args:
            num_samples (int): Number of samples to print
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        print(f"\nSample data ({num_samples} examples):")
        print("-" * 80)
        for i in range(min(num_samples, len(self.dataset))):
            example = self.dataset[i]
            print(f"Example {i + 1}:")
            for key, value in example.items():
                if key == 'image':
                    print(f"  {key}: <PIL.Image object>")
                else:
                    print(f"  {key}: {value}")
            print("-" * 80)


def load_alzheimer_data(
    dataset_name: str = 'Falah/Alzheimer_MRI',
    split: str = 'train',
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Convenience function to load and split Alzheimer's MRI dataset.
    
    Args:
        dataset_name (str): Name of the dataset on Hugging Face Hub
        split (str): Dataset split to load
        test_size (float): Proportion for test set
        val_size (float): Proportion for validation set
        random_state (int): Random seed
    
    Returns:
        Tuple[Dataset, Dataset, Dataset]: train, validation, and test datasets
    """
    loader = AlzheimerDataLoader(dataset_name)
    loader.load_data(split=split)
    train_data, val_data, test_data = loader.split_data(
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
    return train_data, val_data, test_data


if __name__ == '__main__':
    loader = AlzheimerDataLoader()
    
    dataset = loader.load_data(split='train')
    
    info = loader.get_dataset_info()
    print("\nDataset Information:")
    print(f"  Number of examples: {info['num_examples']}")
    print(f"  Features: {info['features']}")
    if 'class_distribution' in info:
        print(f"  Class distribution: {info['class_distribution']}")
    
    loader.print_sample(num_samples=3)
    
    train_data, val_data, test_data = loader.split_data(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )