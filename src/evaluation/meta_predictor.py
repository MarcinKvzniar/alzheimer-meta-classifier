"""
Meta Predictor module for ensemble predictions using multiple trained classifiers.
Combines Random Forest, SVM, and Gradient Boosting classifiers for robust predictions.
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Any
import random

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.random_forest_classifier import RandomForestAlzheimerClassifier
from src.models.svm_classifier import SVMAlzheimerClassifier
from src.models.gradient_boosting_classifier import GradientBoostingAlzheimerClassifier
from src.data.data_loader import load_alzheimer_data
from PIL import Image


class MetaPredictor:
    """
    Meta predictor that combines predictions from multiple classifiers.
    Uses ensemble voting and provides detailed analysis of individual classifier results.
    """
    
    def __init__(self, models_dir: str = None):
        """
        Initialize the meta predictor.
        
        Args:
            models_dir: Directory containing saved model files (.pkl)
        """
        if models_dir is None:
            models_dir = project_root / "models"
        else:
            models_dir = Path(models_dir)
        
        self.models_dir = models_dir
        self.classifiers = {}
        self.class_names = {
            0: "Mild Demented",
            1: "Moderate Demented", 
            2: "Non Demented",
            3: "Very Mild Demented"
        }
        
    def load_classifiers(self):
        """
        Load all trained classifiers from disk.
        """
        print("=" * 80)
        print("Loading Trained Classifiers")
        print("=" * 80)
        
        # Load Random Forest
        rf_path = self.models_dir / "random_forest_model.pkl"
        if rf_path.exists():
            print(f"\nLoading Random Forest classifier from {rf_path}...")
            rf_classifier = RandomForestAlzheimerClassifier()
            rf_classifier.load_model(str(rf_path))
            self.classifiers['Random Forest'] = rf_classifier
            print("✓ Random Forest loaded successfully")
        else:
            print(f"✗ Random Forest model not found at {rf_path}")
        
        # Load SVM
        svm_path = self.models_dir / "svm_model.pkl"
        if svm_path.exists():
            print(f"\nLoading SVM classifier from {svm_path}...")
            svm_classifier = SVMAlzheimerClassifier()
            svm_classifier.load_model(str(svm_path))
            self.classifiers['SVM'] = svm_classifier
            print("✓ SVM loaded successfully")
        else:
            print(f"✗ SVM model not found at {svm_path}")
        
        # Load Gradient Boosting
        gb_path = self.models_dir / "gradient_boosting_model.pkl"
        if gb_path.exists():
            print(f"\nLoading Gradient Boosting classifier from {gb_path}...")
            gb_classifier = GradientBoostingAlzheimerClassifier()
            gb_classifier.load_model(str(gb_path))
            self.classifiers['Gradient Boosting'] = gb_classifier
            print("✓ Gradient Boosting loaded successfully")
        else:
            print(f"✗ Gradient Boosting model not found at {gb_path}")
        
        if not self.classifiers:
            raise ValueError("No classifiers were loaded! Please train models first.")
        
        print(f"\n{'=' * 80}")
        print(f"Successfully loaded {len(self.classifiers)} classifier(s)")
        print("=" * 80)
    
    def predict_single_image(
        self, 
        image: Image.Image, 
        true_label: int = None
    ) -> Dict[str, Any]:
        """
        Predict class for a single image using all classifiers.
        
        Args:
            image: PIL Image to classify
            true_label: Optional true label for comparison
            
        Returns:
            Dictionary containing predictions from all classifiers and ensemble result
        """
        if not self.classifiers:
            raise ValueError("No classifiers loaded. Call load_classifiers() first.")
        
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
        
        # Get predictions from each classifier
        for name, classifier in self.classifiers.items():
            pred = classifier.predict(image)
            proba = classifier.predict_proba(image)
            
            results['individual_predictions'][name] = pred
            results['individual_probabilities'][name] = proba
            
            predictions.append(pred)
            all_probabilities.append(proba)
        
        # Calculate ensemble prediction (majority voting)
        predictions_array = np.array(predictions)
        unique, counts = np.unique(predictions_array, return_counts=True)
        ensemble_pred = unique[np.argmax(counts)]
        
        # Calculate agreement score (percentage of classifiers that agree)
        agreement_count = np.sum(predictions_array == ensemble_pred)
        results['agreement_score'] = agreement_count / len(predictions)
        
        # Calculate ensemble probability (mean of all probabilities)
        mean_probabilities = np.mean(all_probabilities, axis=0)
        results['ensemble_probability'] = mean_probabilities
        results['ensemble_prediction'] = ensemble_pred
        
        return results
    
    def display_prediction_results(
        self, 
        results: Dict[str, Any], 
        image_index: int = None
    ):
        """
        Display formatted prediction results.
        
        Args:
            results: Results dictionary from predict_single_image
            image_index: Optional index of the image for display
        """
        print("\n" + "=" * 80)
        if image_index is not None:
            print(f"PREDICTION RESULTS FOR IMAGE #{image_index}")
        else:
            print("PREDICTION RESULTS")
        print("=" * 80)
        
        # Display true label if available
        if results['true_label'] is not None:
            true_label = results['true_label']
            print(f"\n🎯 TRUE LABEL: {true_label} - {self.class_names.get(true_label, 'Unknown')}")
            print("-" * 80)
        
        # Display individual classifier results
        print("\n📊 INDIVIDUAL CLASSIFIER PREDICTIONS:")
        print("-" * 80)
        
        for name, pred in results['individual_predictions'].items():
            proba = results['individual_probabilities'][name]
            confidence = proba[pred]
            class_name = self.class_names.get(pred, 'Unknown')
            
            # Add checkmark if correct (when true label is available)
            correct_marker = ""
            if results['true_label'] is not None:
                correct_marker = " ✓" if pred == results['true_label'] else " ✗"
            
            print(f"\n{name}:")
            print(f"  Prediction: {pred} - {class_name}{correct_marker}")
            print(f"  Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            print(f"  All probabilities: {proba}")
        
        # Display ensemble results
        print("\n" + "-" * 80)
        print("🤝 ENSEMBLE PREDICTION (Meta-Classifier):")
        print("-" * 80)
        
        ensemble_pred = results['ensemble_prediction']
        ensemble_class = self.class_names.get(ensemble_pred, 'Unknown')
        agreement = results['agreement_score']
        
        # Add checkmark if correct
        correct_marker = ""
        if results['true_label'] is not None:
            correct_marker = " ✓" if ensemble_pred == results['true_label'] else " ✗"
        
        print(f"\nFinal Prediction: {ensemble_pred} - {ensemble_class}{correct_marker}")
        print(f"Ensemble Confidence: {results['ensemble_probability'][ensemble_pred]:.4f}")
        print(f"Agreement Score: {agreement:.2%} ({int(agreement * len(self.classifiers))}/{len(self.classifiers)} classifiers agree)")
        
        # Display agreement level
        if agreement == 1.0:
            agreement_level = "🟢 UNANIMOUS"
        elif agreement >= 0.67:
            agreement_level = "🟡 STRONG MAJORITY"
        elif agreement >= 0.5:
            agreement_level = "🟠 SIMPLE MAJORITY"
        else:
            agreement_level = "🔴 NO CONSENSUS"
        
        print(f"Agreement Level: {agreement_level}")
        
        # Display mean probabilities
        print(f"\nMean Probabilities Across All Classifiers:")
        for label, prob in enumerate(results['ensemble_probability']):
            class_name = self.class_names.get(label, f'Class {label}')
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"  {label} - {class_name:20s}: {bar} {prob:.4f}")
        
        print("\n" + "=" * 80)
    
    def evaluate_random_samples(
        self, 
        dataset, 
        num_samples: int = 5,
        seed: int = None
    ):
        """
        Evaluate the meta predictor on random samples from a dataset.
        
        Args:
            dataset: Dataset to sample from
            num_samples: Number of random samples to evaluate
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        print("\n" + "=" * 80)
        print(f"META-PREDICTOR EVALUATION ON {num_samples} RANDOM SAMPLES")
        print("=" * 80)
        
        # Get random indices
        dataset_size = len(dataset)
        indices = random.sample(range(dataset_size), min(num_samples, dataset_size))
        
        all_results = []
        correct_individual = {name: 0 for name in self.classifiers.keys()}
        correct_ensemble = 0
        
        for i, idx in enumerate(indices, 1):
            example = dataset[idx]
            image = example['image']
            true_label = example['label']
            
            # Get predictions
            results = self.predict_single_image(image, true_label)
            all_results.append(results)
            
            # Display results
            self.display_prediction_results(results, image_index=idx)
            
            # Track accuracy
            for name, pred in results['individual_predictions'].items():
                if pred == true_label:
                    correct_individual[name] += 1
            
            if results['ensemble_prediction'] == true_label:
                correct_ensemble += 1
        
        # Display summary statistics
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        print(f"\nTotal Samples Evaluated: {num_samples}")
        print("\nIndividual Classifier Accuracy:")
        for name, correct in correct_individual.items():
            accuracy = correct / num_samples
            print(f"  {name:20s}: {correct}/{num_samples} = {accuracy:.2%}")
        
        ensemble_accuracy = correct_ensemble / num_samples
        print(f"\nEnsemble Accuracy:        {correct_ensemble}/{num_samples} = {ensemble_accuracy:.2%}")
        
        # Calculate average agreement
        avg_agreement = np.mean([r['agreement_score'] for r in all_results])
        print(f"Average Agreement Score:  {avg_agreement:.2%}")
        
        print("\n" + "=" * 80)
        
        return all_results


def main():
    """
    Main function demonstrating the meta predictor functionality.
    """
    print("\n" + "=" * 80)
    print("META-PREDICTOR: ENSEMBLE ALZHEIMER'S DISEASE CLASSIFICATION")
    print("=" * 80)
    
    # Initialize meta predictor
    meta_predictor = MetaPredictor()
    
    # Load all classifiers
    meta_predictor.load_classifiers()
    
    # Load test data
    print("\n" + "=" * 80)
    print("Loading Test Dataset")
    print("=" * 80)
    
    _, _, test_data = load_alzheimer_data(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    print(f"\nTest dataset size: {len(test_data)} images")
    
    # Evaluate on random samples
    num_samples = 5
    results = meta_predictor.evaluate_random_samples(
        test_data, 
        num_samples=num_samples,
        seed=42
    )
    
    print("\n" + "=" * 80)
    print("META-PREDICTOR DEMONSTRATION COMPLETED!")
    print("=" * 80)
    print("\nThe meta-predictor combines predictions from multiple classifiers to")
    print("provide more robust and reliable predictions through ensemble voting.")
    print("=" * 80)


if __name__ == '__main__':
    main()
