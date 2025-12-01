"""
Random Forest parameter optimization for Alzheimer's disease classification.
Tests different hyperparameter combinations and compares their performance.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.random_forest_classifier import RandomForestAlzheimerClassifier
from src.data.combined_data_loader import load_combined_alzheimer_data
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import numpy as np
import time
from typing import Dict, List, Tuple, Any
import json


class RandomForestOptimizer:
    """
    Optimizer for finding the best Random Forest hyperparameters.
    """

    def __init__(self, train_dataset, val_dataset, test_dataset):
        """
        Initialize optimizer with datasets.

        Args:
            train_dataset: Training dataset (CombinedDataset)
            val_dataset: Validation dataset (CombinedDataset)
            test_dataset: Test dataset (CombinedDataset)
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.results = []

    def define_parameter_grid(self) -> List[Dict[str, Any]]:
        """
        Define different parameter configurations to test.

        Returns:
            List of parameter dictionaries
        """
        param_configs = [
            # Baseline configuration
            {
                'name': 'Baseline',
                'n_estimators': 200,
                'max_depth': None,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            },
            # More trees
            {
                'name': 'More Trees',
                'n_estimators': 500,
                'max_depth': None,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            },
            # Limited depth (prevent overfitting)
            {
                'name': 'Limited Depth',
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            },
            # Conservative (more regularization)
            {
                'name': 'Conservative',
                'n_estimators': 300,
                'max_depth': 8,
                'min_samples_split': 10,
                'min_samples_leaf': 5
            },
            # Aggressive (less regularization)
            {
                'name': 'Aggressive',
                'n_estimators': 150,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            },
            # Fast training (fewer trees, limited depth)
            {
                'name': 'Fast Training',
                'n_estimators': 100,
                'max_depth': 6,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            },
            # High capacity (many trees, moderate depth)
            {
                'name': 'High Capacity',
                'n_estimators': 400,
                'max_depth': 12,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            },
            # Balanced regularization
            {
                'name': 'Balanced',
                'n_estimators': 250,
                'max_depth': 10,
                'min_samples_split': 8,
                'min_samples_leaf': 3
            },
            # Very deep trees
            {
                'name': 'Very Deep',
                'n_estimators': 150,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            },
            # Shallow but wide
            {
                'name': 'Shallow Wide',
                'n_estimators': 600,
                'max_depth': 5,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }
        ]

        return param_configs

    def train_and_evaluate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train a model with given parameters and evaluate it.

        Args:
            params: Dictionary of hyperparameters

        Returns:
            Dictionary containing results
        """
        config_name = params.pop('name')
        print(f"\n{'='*80}")
        print(f"Testing Configuration: {config_name}")
        print(f"Parameters: {params}")
        print(f"{'='*80}")

        # Create and train classifier
        start_time = time.time()
        classifier = RandomForestAlzheimerClassifier(**params, random_state=42, use_scaling=True)

        print("Training model...")
        classifier.train(self.train_dataset)
        training_time = time.time() - start_time

        # Evaluate on validation set
        print("Evaluating on validation set...")
        val_start = time.time()
        y_val_pred = []
        y_val_true = []
        for example in self.val_dataset:
            pred = classifier.predict(example['image'])
            y_val_pred.append(pred)
            y_val_true.append(example['label'])
        val_time = time.time() - val_start

        val_accuracy = accuracy_score(y_val_true, y_val_pred)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            y_val_true, y_val_pred, average='weighted', zero_division=0
        )

        # Evaluate on test set
        print("Evaluating on test set...")
        test_start = time.time()
        y_test_pred = []
        y_test_true = []
        for example in self.test_dataset:
            pred = classifier.predict(example['image'])
            y_test_pred.append(pred)
            y_test_true.append(example['label'])
        test_time = time.time() - test_start

        test_accuracy = accuracy_score(y_test_true, y_test_pred)
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            y_test_true, y_test_pred, average='weighted', zero_division=0
        )

        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
            y_test_true, y_test_pred, average=None, zero_division=0
        )

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test_true, y_test_pred)

        # Print results
        print(f"\nValidation Results:")
        print(f"  Accuracy:  {val_accuracy:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall:    {val_recall:.4f}")
        print(f"  F1-Score:  {val_f1:.4f}")

        print(f"\nTest Results:")
        print(f"  Accuracy:  {test_accuracy:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall:    {test_recall:.4f}")
        print(f"  F1-Score:  {test_f1:.4f}")

        print(f"\nTiming:")
        print(f"  Training time:   {training_time:.2f}s")
        print(f"  Val pred time:   {val_time:.2f}s")
        print(f"  Test pred time:  {test_time:.2f}s")

        # Store results
        result = {
            'name': config_name,
            'parameters': params,
            'validation': {
                'accuracy': float(val_accuracy),
                'precision': float(val_precision),
                'recall': float(val_recall),
                'f1_score': float(val_f1)
            },
            'test': {
                'accuracy': float(test_accuracy),
                'precision': float(test_precision),
                'recall': float(test_recall),
                'f1_score': float(test_f1),
                'per_class_metrics': {
                    'precision': per_class_precision.tolist(),
                    'recall': per_class_recall.tolist(),
                    'f1_score': per_class_f1.tolist(),
                    'support': support.tolist()
                },
                'confusion_matrix': conf_matrix.tolist()
            },
            'timing': {
                'training_time': float(training_time),
                'val_prediction_time': float(val_time),
                'test_prediction_time': float(test_time)
            }
        }

        return result

    def optimize(self) -> List[Dict[str, Any]]:
        """
        Run optimization by testing all parameter configurations.

        Returns:
            List of results for all configurations
        """
        param_configs = self.define_parameter_grid()

        print(f"\n{'#'*80}")
        print(f"# Starting Random Forest Optimization")
        print(f"# Testing {len(param_configs)} different configurations")
        print(f"# Dataset sizes: Train={len(self.train_dataset)}, Val={len(self.val_dataset)}, Test={len(self.test_dataset)}")
        print(f"{'#'*80}\n")

        for config in param_configs:
            try:
                result = self.train_and_evaluate(config)
                self.results.append(result)
            except Exception as e:
                print(f"Error with configuration {config['name']}: {str(e)}")
                continue

        return self.results

    def print_summary(self):
        """Print a summary comparison of all tested configurations."""
        if not self.results:
            print("No results to summarize.")
            return

        print(f"\n{'='*100}")
        print(f"OPTIMIZATION SUMMARY - COMPARISON OF ALL CONFIGURATIONS")
        print(f"{'='*100}\n")

        # Sort by validation F1 score
        sorted_results = sorted(
            self.results,
            key=lambda x: x['validation']['f1_score'],
            reverse=True
        )

        # Print header
        print(f"{'Rank':<6} {'Configuration':<25} {'Val Acc':<10} {'Val F1':<10} {'Test Acc':<10} {'Test F1':<10} {'Train Time':<12}")
        print(f"{'-'*100}")

        # Print each configuration
        for idx, result in enumerate(sorted_results, 1):
            name = result['name']
            val_acc = result['validation']['accuracy']
            val_f1 = result['validation']['f1_score']
            test_acc = result['test']['accuracy']
            test_f1 = result['test']['f1_score']
            train_time = result['timing']['training_time']

            marker = "★" if idx == 1 else " "
            print(f"{marker} {idx:<4} {name:<25} {val_acc:<10.4f} {val_f1:<10.4f} {test_acc:<10.4f} {test_f1:<10.4f} {train_time:<12.2f}s")

        print(f"\n{'='*100}")

        # Best configuration details
        best = sorted_results[0]
        print(f"\n🏆 BEST CONFIGURATION: {best['name']}")
        print(f"\nParameters:")
        for param, value in best['parameters'].items():
            print(f"  {param}: {value}")

        print(f"\nValidation Metrics:")
        print(f"  Accuracy:  {best['validation']['accuracy']:.4f}")
        print(f"  Precision: {best['validation']['precision']:.4f}")
        print(f"  Recall:    {best['validation']['recall']:.4f}")
        print(f"  F1-Score:  {best['validation']['f1_score']:.4f}")

        print(f"\nTest Metrics:")
        print(f"  Accuracy:  {best['test']['accuracy']:.4f}")
        print(f"  Precision: {best['test']['precision']:.4f}")
        print(f"  Recall:    {best['test']['recall']:.4f}")
        print(f"  F1-Score:  {best['test']['f1_score']:.4f}")

        print(f"\nTraining Time: {best['timing']['training_time']:.2f}s")

        print(f"\n{'='*100}\n")

    def save_results(self, filepath: str = "optimization_results.json"):
        """
        Save optimization results to a JSON file.

        Args:
            filepath: Path to save results
        """
        output_path = Path(filepath)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to: {output_path.absolute()}")


def main():
    """Main function to run the optimization."""
    print("Loading Alzheimer's disease dataset...")

    # Load data
    train_data, val_data, test_data = load_combined_alzheimer_data(
        use_huggingface=True,
        use_local=True,
        val_size=0.15,
        test_size=0.15,
        random_state=42
    )

    print(f"\nDataset loaded:")
    print(f"  Training samples:   {len(train_data)}")
    print(f"  Validation samples: {len(val_data)}")
    print(f"  Test samples:       {len(test_data)}")

    # Get class distribution from training data
    labels = [example['label'] for example in train_data]
    print(f"  Classes: {np.unique(labels)}")

    # Create optimizer and run
    optimizer = RandomForestOptimizer(
        train_data, val_data, test_data
    )

    # Run optimization
    results = optimizer.optimize()

    # Print summary
    optimizer.print_summary()

    # Save results
    optimizer.save_results("random_forest_optimization_results.json")

    print("\nOptimization complete!")


if __name__ == "__main__":
    main()