"""
Gradient Boosting classifier for Alzheimer's disease classification.
Uses image features with scikit-learn's GradientBoostingClassifier.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.combined_data_loader import load_combined_alzheimer_data
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from PIL import Image
import pickle


class GradientBoostingAlzheimerClassifier:
    """
    Gradient Boosting classifier for Alzheimer's disease classification.
    Extracts rich image features and uses Gradient Boosting for classification.
    """
    
    def __init__(
        self, 
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        min_samples_split: int = 4,
        min_samples_leaf: int = 2,
        subsample: float = 0.8,
        random_state: int = 42,
        use_scaling: bool = True
    ):
        """
        Initialize the Gradient Boosting classifier.
        
        Args:
            n_estimators: Number of boosting stages
            learning_rate: Learning rate shrinks contribution of each tree
            max_depth: Maximum depth of individual regression estimators
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at leaf node
            subsample: Fraction of samples used for fitting base learners
            random_state: Random seed for reproducibility
            use_scaling: Whether to use feature scaling
        """
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=random_state,
            verbose=0
        )
        self.random_state = random_state
        self.use_scaling = use_scaling
        self.scaler = StandardScaler() if use_scaling else None
        self.is_trained = False
        self.training_history = []
        
    def _extract_features(self, image: Image.Image) -> np.ndarray:
        """
        Extract comprehensive features from an image.
        Includes statistical, spatial, texture, and frequency domain features.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Feature vector as numpy array
        """
        # Resize image for consistent feature extraction
        img_resized = np.array(image.resize((96, 96)))
        
        # Handle grayscale images
        if len(img_resized.shape) == 2:
            img_resized = np.stack([img_resized] * 3, axis=-1)
        
        features = []
        
        # === Global Statistics ===
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
            img_resized.var(),
        ])
        
        # === Per-Channel Statistics ===
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
        
        # === Convert to grayscale for texture analysis ===
        gray = np.mean(img_resized, axis=2).astype(np.float32)
        
        # === Gradient Features ===
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_direction = np.arctan2(grad_y, grad_x)
        
        features.extend([
            grad_magnitude.mean(),
            grad_magnitude.std(),
            grad_magnitude.max(),
            grad_magnitude.min(),
            np.median(grad_magnitude),
        ])
        
        # === Edge Density ===
        edge_threshold = grad_magnitude.mean() + grad_magnitude.std()
        edge_density = (grad_magnitude > edge_threshold).sum() / grad_magnitude.size
        features.append(edge_density)
        
        # === Histogram Features (multiple bins) ===
        hist, _ = np.histogram(img_resized.flatten(), bins=20, range=(0, 256))
        hist_normalized = hist / hist.sum()
        features.extend(hist_normalized.tolist())
        
        # === Entropy ===
        hist_prob = hist_normalized[hist_normalized > 0]
        entropy = -np.sum(hist_prob * np.log2(hist_prob))
        features.append(entropy)
        
        # === Local Statistics (divide into regions) ===
        h, w = gray.shape
        h_third, w_third = h // 3, w // 3
        
        # 3x3 grid
        for i in range(3):
            for j in range(3):
                region = gray[i*h_third:(i+1)*h_third, j*w_third:(j+1)*w_third]
                features.extend([
                    region.mean(),
                    region.std(),
                ])
        
        # === Contrast and Brightness ===
        contrast = img_resized.std()
        brightness = img_resized.mean()
        features.extend([contrast, brightness])
        
        # === Higher-order statistics ===
        from scipy import stats
        flattened = img_resized.flatten()
        features.extend([
            stats.skew(flattened),
            stats.kurtosis(flattened)
        ])
        
        # === Texture patterns (Local Binary Pattern approximation) ===
        # Simple directional differences
        diff_horizontal = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
        diff_vertical = np.abs(gray[1:, :] - gray[:-1, :]).mean()
        diff_diag1 = np.abs(gray[1:, 1:] - gray[:-1, :-1]).mean()
        diff_diag2 = np.abs(gray[1:, :-1] - gray[:-1, 1:]).mean()
        
        features.extend([
            diff_horizontal,
            diff_vertical,
            diff_diag1,
            diff_diag2
        ])
        
        # === Symmetry features ===
        left_half = gray[:, :w//2]
        right_half = np.fliplr(gray[:, w//2:])
        min_width = min(left_half.shape[1], right_half.shape[1])
        symmetry_score = np.mean(np.abs(left_half[:, :min_width] - right_half[:, :min_width]))
        features.append(symmetry_score)
        
        # === Frequency domain features (simple) ===
        fft = np.fft.fft2(gray)
        fft_magnitude = np.abs(fft)
        fft_magnitude_shifted = np.fft.fftshift(fft_magnitude)
        
        features.extend([
            fft_magnitude.mean(),
            fft_magnitude.std(),
            fft_magnitude_shifted[h//2, w//2],  # DC component
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
    
    def train(self, train_dataset, val_dataset=None, verbose: bool = True) -> Dict[str, Any]:
        """
        Train the Gradient Boosting classifier.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            verbose: Whether to print detailed training info
            
        Returns:
            Dictionary with training metrics
        """
        print("=" * 80)
        print("Training Gradient Boosting Classifier")
        print("=" * 80)
        
        # Extract features
        X_train, y_train = self._prepare_features(train_dataset)
        
        # Apply scaling if enabled
        if self.use_scaling:
            print("\nApplying feature scaling...")
            X_train = self.scaler.fit_transform(X_train)
        
        # Train model
        print("\nTraining Gradient Boosting model...")
        print(f"  - Number of estimators: {self.model.n_estimators}")
        print(f"  - Learning rate: {self.model.learning_rate}")
        print(f"  - Max depth: {self.model.max_depth}")
        print(f"  - Subsample: {self.model.subsample}")
        print(f"  - Feature dimension: {X_train.shape[1]}")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("Training completed!")
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        print(f"\nTraining Accuracy: {train_accuracy:.4f}")
        
        # Get training deviance
        self.training_history = self.model.train_score_.tolist()
        if verbose:
            print(f"Final training deviance: {self.training_history[-1]:.4f}")
        
        results = {
            'train_accuracy': train_accuracy,
            'training_history': self.training_history
        }
        
        # Evaluate on validation set if provided
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
        
        # Extract features
        X_test, y_test = self._prepare_features(test_dataset)
        
        # Apply scaling if enabled
        if self.use_scaling:
            X_test = self.scaler.transform(X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        # Calculate AUC
        try:
            n_classes = len(np.unique(y_test))
            if n_classes == 2:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        except Exception:
            auc = None
        
        # Print results
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
        
        # Feature importance
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
    
    def get_staged_predictions(self, image: Image.Image, stages: Optional[List[int]] = None) -> np.ndarray:
        """
        Get predictions at different boosting stages.
        Useful for analyzing how predictions evolve during boosting.
        
        Args:
            image: Input PIL Image
            stages: List of stages to get predictions for (None = all stages)
            
        Returns:
            Array of predictions at each stage
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        features = self._extract_features(image).reshape(1, -1)
        
        if self.use_scaling:
            features = self.scaler.transform(features)
        
        staged_preds = []
        for pred in self.model.staged_predict(features):
            staged_preds.append(pred[0])
        
        if stages is not None:
            return np.array([staged_preds[s] for s in stages if s < len(staged_preds)])
        
        return np.array(staged_preds)
    
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
            'is_trained': self.is_trained,
            'training_history': self.training_history
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
        self.training_history = model_data.get('training_history', [])
        
        print(f"Model loaded from {filepath}")


def main():
    """
    Main function to train and evaluate Gradient Boosting classifier.
    """
    print("\n" + "=" * 80)
    print("Gradient Boosting Classifier for Alzheimer's Disease Classification")
    print("=" * 80)
    
    # Load data (combined HuggingFace + Local)
    print("\nLoading combined dataset (HuggingFace + Local)...")
    train_data, val_data, test_data = load_combined_alzheimer_data(
        use_huggingface=True,
        use_local=True,
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # Create classifier
    print("\nInitializing Gradient Boosting classifier...")
    classifier = GradientBoostingAlzheimerClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42,
        use_scaling=True
    )
    
    # Use subset for faster testing (remove for full training)
    print("\nNote: Using subset of data for demonstration...")
    train_subset = train_data.select(range(min(1000, len(train_data))))
    val_subset = val_data.select(range(min(200, len(val_data))))
    test_subset = test_data.select(range(min(200, len(test_data))))
    
    # Train
    train_results = classifier.train(train_subset, val_subset)
    
    # Evaluate on test set
    print("\n" + "=" * 80)
    print("Test Set Evaluation")
    print("=" * 80)
    test_metrics = classifier.evaluate(test_subset, dataset_name="Test")
    
    # Test single prediction
    print("\n" + "=" * 80)
    print("Single Image Prediction Test")
    print("=" * 80)
    sample_image = test_subset[0]['image']
    sample_label = test_subset[0]['label']
    
    prediction = classifier.predict(sample_image)
    probabilities = classifier.predict_proba(sample_image)
    
    print(f"\nTrue Label:      {sample_label}")
    print(f"Predicted Label: {prediction}")
    print(f"Class Probabilities: {probabilities}")
    print(f"Prediction Confidence: {probabilities[prediction]:.4f}")
    
    # Test staged predictions
    print("\n" + "=" * 80)
    print("Staged Predictions Analysis")
    print("=" * 80)
    stages_to_check = [9, 24, 49, 74, 99]  # Check at 10, 25, 50, 75, 100 trees
    staged_preds = classifier.get_staged_predictions(sample_image, stages_to_check)
    
    print("\nPredictions at different boosting stages:")
    for stage, pred in zip(stages_to_check, staged_preds):
        print(f"  After {stage+1} trees: {pred}")
    
    # Save model
    model_path = project_root / "models" / "gradient_boosting_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(str(model_path))
    
    print("\n" + "=" * 80)
    print("Training and Evaluation Completed Successfully!")
    print("=" * 80)


if __name__ == '__main__':
    main()
