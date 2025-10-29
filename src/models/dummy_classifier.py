"""
Example classifier module demonstrating how to use the data loader
and implement a basic classifier for Alzheimer's disease classification.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import AlzheimerDataLoader, load_alzheimer_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
from typing import Tuple, Any
from PIL import Image


class DummyAlzheimerClassifier:
    """
    A dummy classifier for Alzheimer's disease classification.
    Uses simple image statistics (mean pixel values) as features with RandomForest.
    An example to demonstrate the data loading pipeline.
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize the dummy classifier.
        
        Args:
            n_estimators (int): Number of trees in the random forest
            random_state (int): Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        self.random_state = random_state
        
    def _extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract simple features from an image.
        For this dummy classifier, we just compute mean RGB values and basic statistics.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            np.ndarray: Feature vector
        """
        img_array = np.array(image)
        
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        
        img_resized = np.array(image.resize((64, 64)))
        if len(img_resized.shape) == 2:
            img_resized = np.stack([img_resized] * 3, axis=-1)
        
        features = [
            img_resized.mean(),           # Overall mean
            img_resized.std(),            # Overall std
            img_resized.min(),            # Min value
            img_resized.max(),            # Max value
            img_resized[:, :, 0].mean(),  # Red channel mean
            img_resized[:, :, 1].mean(),  # Green channel mean
            img_resized[:, :, 2].mean(),  # Blue channel mean
            img_resized[:, :, 0].std(),   # Red channel std
            img_resized[:, :, 1].std(),   # Green channel std
            img_resized[:, :, 2].std(),   # Blue channel std
        ]
        
        return np.array(features)
    
    def _prepare_features(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels from dataset.
        
        Args:
            dataset: HuggingFace dataset with 'image' and 'label' columns
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features (X) and labels (y)
        """
        print(f"Extracting features from {len(dataset)} images...")
        
        X = []
        y = []
        
        for i, example in enumerate(dataset):
            if i % 500 == 0:
                print(f"  Processed {i}/{len(dataset)} images...")
            
            features = self._extract_features(example['image'])
            X.append(features)
            y.append(example['label'])
        
        print(f"Feature extraction completed!")
        return np.array(X), np.array(y)
    
    def train(self, train_dataset, val_dataset=None):
        """
        Train the classifier.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset for evaluation
        """
        print("Training Dummy Classifier")

        X_train, y_train = self._prepare_features(train_dataset)
        
        print("\nTraining Random Forest classifier...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
        
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"\nTraining Accuracy: {train_accuracy:.4f}")
        
        if val_dataset is not None:
            print("\nEvaluating on validation set...")
            val_accuracy = self.evaluate(val_dataset)
            print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    def evaluate(self, test_dataset) -> float:
        """
        Evaluate the classifier on test data.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            float: Accuracy score
        """
        X_test, y_test = self._prepare_features(test_dataset)
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print("Evaluation Results")
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy
    
    def predict(self, image: Image.Image) -> int:
        """
        Predict the class for a single image.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            int: Predicted class label
        """
        features = self._extract_features(image).reshape(1, -1)
        return self.model.predict(features)[0]
    
    def predict_proba(self, image: Image.Image) -> np.ndarray:
        """
        Predict class probabilities for a single image.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            np.ndarray: Class probabilities
        """
        features = self._extract_features(image).reshape(1, -1)
        return self.model.predict_proba(features)[0]


def test_data_loader():
    """
    Test the data loader functionality.
    """
    print("Testing data loader")
    
    print("Test 1: Using AlzheimerDataLoader class")
    
    loader = AlzheimerDataLoader()
    dataset = loader.load_data(split='train')
    
    info = loader.get_dataset_info()
    print("\nDataset Information:")
    print(f"  Number of examples: {info['num_examples']}")
    print(f"  Column names: {info['column_names']}")
    if 'class_distribution' in info:
        print(f"  Class distribution: {info['class_distribution']}")
    
    loader.print_sample(num_samples=2)
    
    print("\nSplitting data...")
    train_data, val_data, test_data = loader.split_data(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    print("Test 2: Using convenience function")
    
    train_data2, val_data2, test_data2 = load_alzheimer_data(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    print("\nData loader tests completed successfully")
    
    return train_data, val_data, test_data


def main():
    """
    Main function to test data loader and train dummy classifier.
    """
    print("Dummy classifier test")
    
    train_data, val_data, test_data = test_data_loader()
    
    print("\n\nCreating dummy classifier...")
    classifier = DummyAlzheimerClassifier(n_estimators=50, random_state=42)
    
    print("\nNote: Using subset of data for quick testing...")
    train_subset = train_data.select(range(min(500, len(train_data))))
    val_subset = val_data.select(range(min(100, len(val_data))))
    test_subset = test_data.select(range(min(100, len(test_data))))
    
    classifier.train(train_subset, val_subset)
    
    print("\n\nEvaluating on test set...")
    test_accuracy = classifier.evaluate(test_subset)
    
    print("Testing single prediction")
    sample_image = test_subset[0]['image']
    sample_label = test_subset[0]['label']
    
    prediction = classifier.predict(sample_image)
    probabilities = classifier.predict_proba(sample_image)
    
    print(f"\nSample Image True Label: {sample_label}")
    print(f"Predicted Label: {prediction}")
    print(f"Class Probabilities: {probabilities}")
    
    print("All tests completed successfully")


if __name__ == '__main__':
    main()
