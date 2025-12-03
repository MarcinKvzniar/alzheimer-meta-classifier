"""
Random Forest classifier for Alzheimer's disease classification.
Uses image features with scikit-learn's RandomForestClassifier.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.combined_data_loader import load_combined_alzheimer_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Tuple, Dict, Any, Optional
from PIL import Image
import pickle


class RandomForestAlzheimerClassifier:
    """
    Random Forest classifier for Alzheimer's disease classification.
    Extracts advanced image features and uses Random Forest for classification.
    """
    
    def __init__(
        self, 
        n_estimators: int = 200, 
        max_depth: Optional[int] = None,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        use_scaling: bool = True
    ):
        """
        Initialize the Random Forest classifier.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at leaf node
            random_state: Random seed for reproducibility
            use_scaling: Whether to use feature scaling
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.random_state = random_state
        self.use_scaling = use_scaling
        self.scaler = StandardScaler() if use_scaling else None
        self.is_trained = False
        
    def _extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract comprehensive features from an image.
        Includes statistical features, texture features, and histogram features.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Feature vector as numpy array
        """
        img_resized = np.array(image.resize((128, 128)))
        
        if len(img_resized.shape) == 2:
            img_resized = np.stack([img_resized] * 3, axis=-1)
        
        features = []
        
        features.extend([
            img_resized.mean(),
            img_resized.std(),
            img_resized.min(),
            img_resized.max(),
            np.median(img_resized),
            np.percentile(img_resized, 25),
            np.percentile(img_resized, 75),
        ])
        
        for channel in range(3):
            channel_data = img_resized[:, :, channel]
            features.extend([
                channel_data.mean(),
                channel_data.std(),
                channel_data.min(),
                channel_data.max(),
                np.median(channel_data),
            ])
        
        gray = np.mean(img_resized, axis=2).astype(np.float32)
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            grad_magnitude.mean(),
            grad_magnitude.std(),
            grad_magnitude.max(),
        ])
        
        hist, _ = np.histogram(img_resized.flatten(), bins=10, range=(0, 256))
        hist_normalized = hist / hist.sum()
        features.extend(hist_normalized.tolist())
        
        hist_prob = hist_normalized[hist_normalized > 0]
        entropy = -np.sum(hist_prob * np.log2(hist_prob))
        features.append(entropy)
        
        contrast = img_resized.std()
        brightness = img_resized.mean()
        features.extend([contrast, brightness])
        
        return np.array(features)
    
    def _prepare_features(self, dataset) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and labels from dataset.
        
        Args:
            dataset: HuggingFace dataset with 'image' and 'label' columns
            
        Returns:
            Tuple of features (X) and labels (y)
        """
        print(f"Extracting features from {len(dataset)} images...")
        
        X = []
        y = []
        
        for i, example in enumerate(dataset):
            if i % 500 == 0 and i > 0:
                print(f"  Processed {i}/{len(dataset)} images...")
            
            features = self._extract_features(example['image'])
            X.append(features)
            y.append(example['label'])
        
        print(f"Feature extraction completed!")
        return np.array(X), np.array(y)
    
    def train(self, train_dataset, val_dataset=None) -> Dict[str, Any]:
        """
        Train the Random Forest classifier.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            
        Returns:
            Dictionary with training metrics
        """
        print("=" * 80)
        print("Training Random Forest Classifier")
        print("=" * 80)
        
        X_train, y_train = self._prepare_features(train_dataset)
        
        if self.use_scaling:
            print("\nApplying feature scaling...")
            X_train = self.scaler.fit_transform(X_train)
        
        print("\nTraining Random Forest model...")
        print(f"  - Number of trees: {self.model.n_estimators}")
        print(f"  - Max depth: {self.model.max_depth}")
        print(f"  - Feature dimension: {X_train.shape[1]}")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training completed!")
        
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"\nTraining Accuracy: {train_accuracy:.4f}")
        
        results = {'train_accuracy': train_accuracy}
        
        if val_dataset is not None:
            print("\n" + "=" * 80)
            print("Validation Evaluation")
            print("=" * 80)
            val_metrics = self.evaluate(val_dataset, dataset_name="Validation")
            results['val_metrics'] = val_metrics
        
        return results
    
    def evaluate(self, test_dataset, dataset_name: str = "Test") -> Dict[str, Any]:
        """
        Evaluate the classifier on test data.
        
        Args:
            test_dataset: Test dataset
            dataset_name: Name of the dataset for logging
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_test, y_test = self._prepare_features(test_dataset)
        
        if self.use_scaling:
            X_test = self.scaler.transform(X_test)
        
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        try:
            n_classes = len(np.unique(y_test))
            if n_classes == 2:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        except Exception:
            auc = None
        
        print(f"\n{dataset_name} Set Evaluation Results")
        print("-" * 80)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        if auc is not None:
            print(f"AUC:       {auc:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        feature_importance = self.model.feature_importances_
        top_k = 10
        top_indices = np.argsort(feature_importance)[-top_k:][::-1]
        
        print(f"\nTop {top_k} Most Important Features:")
        for i, idx in enumerate(top_indices):
            print(f"  {i+1}. Feature {idx}: {feature_importance[idx]:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'feature_importance': feature_importance.tolist()
        }
    
    def predict(self, image: Image.Image) -> int:
        """
        Predict the class for a single image.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Predicted class label
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        features = self._extract_features(image).reshape(1, -1)
        
        if self.use_scaling:
            features = self.scaler.transform(features)
        
        return self.model.predict(features)[0]
    
    def predict_proba(self, image: Image.Image) -> np.ndarray:
        """
        Predict class probabilities for a single image.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Class probabilities array
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        features = self._extract_features(image).reshape(1, -1)
        
        if self.use_scaling:
            features = self.scaler.transform(features)
        
        return self.model.predict_proba(features)[0]
    
    def save_model(self, filepath: str):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'use_scaling': self.use_scaling,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.use_scaling = model_data['use_scaling']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")


def main():
    """
    Main function to train and evaluate Random Forest classifier.
    """
    print("\n" + "=" * 80)
    print("Random Forest Classifier for Alzheimer's Disease Classification")
    print("=" * 80)
    
    print("\nLoading combined dataset (HuggingFace + Local)...")
    train_data, val_data, test_data = load_combined_alzheimer_data(
        use_huggingface=True,
        use_local=True,
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    print("\nInitializing Random Forest classifier...")
    classifier = RandomForestAlzheimerClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        use_scaling=True
    )
    
    
    train_results = classifier.train(train_data, val_data)
    
    print("\n" + "=" * 80)
    print("Test Set Evaluation")
    print("=" * 80)
    test_metrics = classifier.evaluate(test_data, dataset_name="Test")
    
    print("\n" + "=" * 80)
    print("Single Image Prediction Test")
    print("=" * 80)
    sample_image = test_data[0]['image']
    sample_label = test_data[0]['label']
    
    prediction = classifier.predict(sample_image)
    probabilities = classifier.predict_proba(sample_image)
    
    print(f"\nTrue Label:      {sample_label}")
    print(f"Predicted Label: {prediction}")
    print(f"Class Probabilities: {probabilities}")
    print(f"Prediction Confidence: {probabilities[prediction]:.4f}")
    
    model_path = project_root / "models" / "random_forest_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(str(model_path))
    
    print("\n" + "=" * 80)
    print("Training and Evaluation Completed Successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
