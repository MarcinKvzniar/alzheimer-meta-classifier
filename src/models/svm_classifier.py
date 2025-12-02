"""
Support Vector Machine classifier for Alzheimer's disease classification.
Uses image features with scikit-learn's SVC.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.combined_data_loader import load_combined_alzheimer_data
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from typing import Tuple, Dict, Any, Optional
from PIL import Image
import pickle


class SVMAlzheimerClassifier:
    """
    Support Vector Machine classifier for Alzheimer's disease classification.
    Extracts image features, applies PCA for dimensionality reduction, and uses SVM for classification.
    """
    
    def __init__(
        self, 
        kernel: str = 'rbf',
        C: float = 1.0,
        gamma: str = 'scale',
        use_pca: bool = True,
        n_components: int = 50,
        random_state: int = 42
    ):
        """
        Initialize the SVM classifier.
        
        Args:
            kernel: SVM kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'
            use_pca: Whether to use PCA for dimensionality reduction
            n_components: Number of PCA components
            random_state: Random seed for reproducibility
        """
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=random_state,
            probability=True,
            class_weight='balanced'
        )
        self.random_state = random_state
        self.use_pca = use_pca
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=random_state) if use_pca else None
        self.is_trained = False
        
    def _extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract comprehensive features from an image.
        Includes spatial, statistical, and texture features.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Feature vector as numpy array
        """
        img_resized = np.array(image.resize((64, 64)))
        
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
            np.percentile(img_resized, 10),
            np.percentile(img_resized, 90),
        ])
        
        for channel in range(3):
            channel_data = img_resized[:, :, channel]
            features.extend([
                channel_data.mean(),
                channel_data.std(),
                channel_data.min(),
                channel_data.max(),
                np.median(channel_data),
                channel_data.var(),
            ])
        
        gray = np.mean(img_resized, axis=2).astype(np.float32)
        
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            grad_magnitude.mean(),
            grad_magnitude.std(),
            grad_magnitude.max(),
            grad_magnitude.min(),
        ])
        
        edge_threshold = grad_magnitude.mean() + grad_magnitude.std()
        edge_density = (grad_magnitude > edge_threshold).sum() / grad_magnitude.size
        features.append(edge_density)
        
        hist, _ = np.histogram(img_resized.flatten(), bins=16, range=(0, 256))
        hist_normalized = hist / hist.sum()
        features.extend(hist_normalized.tolist())
        
        hist_prob = hist_normalized[hist_normalized > 0]
        entropy = -np.sum(hist_prob * np.log2(hist_prob))
        features.append(entropy)
        
        h, w = gray.shape
        h_mid, w_mid = h // 2, w // 2
        quadrants = [
            gray[:h_mid, :w_mid],
            gray[:h_mid, w_mid:],
            gray[h_mid:, :w_mid],
            gray[h_mid:, w_mid:]
        ]
        
        for quadrant in quadrants:
            features.extend([
                quadrant.mean(),
                quadrant.std(),
            ])
        
        contrast = img_resized.std()
        brightness = img_resized.mean()
        features.extend([contrast, brightness])
        
        from scipy import stats
        flattened = img_resized.flatten()
        features.extend([
            stats.skew(flattened),
            stats.kurtosis(flattened)
        ])
        
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
        Train the SVM classifier.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            
        Returns:
            Dictionary with training metrics
        """
        print("=" * 80)
        print("Training SVM Classifier")
        print("=" * 80)
        
        X_train, y_train = self._prepare_features(train_dataset)
        
        print("\nApplying feature scaling...")
        X_train = self.scaler.fit_transform(X_train)
        
        if self.use_pca:
            print(f"\nApplying PCA (reducing to {self.n_components} components)...")
            print(f"  Original feature dimension: {X_train.shape[1]}")
            X_train = self.pca.fit_transform(X_train)
            print(f"  Reduced feature dimension: {X_train.shape[1]}")
            
            explained_variance = self.pca.explained_variance_ratio_.sum()
            print(f"  Explained variance: {explained_variance:.4f}")
        
        print("\nTraining SVM model...")
        print(f"  - Kernel: {self.model.kernel}")
        print(f"  - C: {self.model.C}")
        print(f"  - Gamma: {self.model.gamma}")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training completed!")
        
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"\nTraining Accuracy: {train_accuracy:.4f}")
        
        print(f"Number of support vectors: {self.model.n_support_}")
        
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
        
        X_test = self.scaler.transform(X_test)
        if self.use_pca:
            X_test = self.pca.transform(X_test)
        
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
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
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
        features = self.scaler.transform(features)
        
        if self.use_pca:
            features = self.pca.transform(features)
        
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
        features = self.scaler.transform(features)
        
        if self.use_pca:
            features = self.pca.transform(features)
        
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
            'pca': self.pca,
            'use_pca': self.use_pca,
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
        self.pca = model_data['pca']
        self.use_pca = model_data['use_pca']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")


def main():
    """
    Main function to train and evaluate SVM classifier.
    """
    print("\n" + "=" * 80)
    print("SVM Classifier for Alzheimer's Disease Classification")
    print("=" * 80)
    
    print("\nLoading combined dataset (HuggingFace + Local)...")
    train_data, val_data, test_data = load_combined_alzheimer_data(
        use_huggingface=True,
        use_local=True,
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    print("\nInitializing SVM classifier...")
    classifier = SVMAlzheimerClassifier(
        kernel='rbf',
        C=1.0,
        gamma=0.1,
        use_pca=True,
        n_components=50,
        random_state=42
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
    
    model_path = project_root / "models" / "svm_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(str(model_path))
    
    print("\n" + "=" * 80)
    print("Training and Evaluation Completed Successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
