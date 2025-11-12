#!/usr/bin/env python3
"""
Turbofan Autoencoder Evaluation Script

STUDENTS: You will complete 3 TODOs to evaluate your trained model.

What you'll learn:
- Loading models from MLflow
- Calculating evaluation metrics
- Logging results back to MLflow

Author: Feda Almuhisen
Course: M2 SID - Processus Data
Institution: Aix-Marseille University
Year: 2025-2026
"""

import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Tuple, Dict, List
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    precision_recall_fscore_support,
    roc_curve,
    roc_auc_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow imports
import mlflow
import mlflow.pytorch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.autoencoder import TurbofanAutoencoder
from data.data_loader import prepare_data_for_training, load_processed_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TurbofanEvaluator:
    """
    Evaluator class for turbofan autoencoder.

    This class handles model evaluation and creates visualizations.
    """

    def __init__(
        self,
        model: TurbofanAutoencoder,
        test_loader: DataLoader,
        device: str = 'cpu'
    ):
        """
        Initialize evaluator.

        Args:
            model: Trained TurbofanAutoencoder
            test_loader: Test DataLoader
            device: Device to evaluate on
        """
        self.model = model.to(device)
        self.model.eval()
        self.test_loader = test_loader
        self.device = device

        logger.info(f"Initialized evaluator on device: {device}")

    def calculate_reconstruction_errors(self) -> np.ndarray:
        """
        Calculate reconstruction error for all test samples.

        Returns:
            Array of reconstruction errors (one per sample)
        """
        logger.info("Calculating reconstruction errors...")

        errors = []

        with torch.no_grad():
            for batch_x, _ in self.test_loader:
                batch_x = batch_x.to(self.device)

                # ==========================================================
                # TODO 1: Calculate reconstruction errors
                # ==========================================================
                # HINT: Use self.model.get_reconstruction_error(batch_x, reduction='none')
                #
                # This method is already implemented in the autoencoder class.
                # It returns the reconstruction error for each sample in the batch.
                #
                # YOUR CODE HERE (1 line):
                batch_errors = self.model.get_reconstruction_error(batch_x, reduction='none')


                errors.append(batch_errors.cpu().numpy())

        errors = np.concatenate(errors)
        logger.info(f" Calculated {len(errors)} reconstruction errors")

        return errors

    def determine_threshold(
        self,
        errors: np.ndarray,
        method: str = 'percentile',
        percentile: float = 95.0
    ) -> float:
        """
        Determine anomaly threshold.

        Args:
            errors: Array of reconstruction errors
            method: Threshold method
            percentile: Percentile to use

        Returns:
            Threshold value
        """
        if method == 'percentile':
            threshold = np.percentile(errors, percentile)
            logger.info(f"Threshold (percentile={percentile}): {threshold:.4f}")
        else:
            threshold = np.mean(errors) + 2 * np.std(errors)
            logger.info(f"Threshold (mean + 2*std): {threshold:.4f}")

        return float(threshold)

    def create_labels_from_rul(
        self,
        test_df: pd.DataFrame,
        rul_threshold: int = 30
    ) -> np.ndarray:
        """
        Create binary anomaly labels from RUL values.

        Assumption: Engines with RUL < threshold are "anomalous"

        Args:
            test_df: Test DataFrame with RUL column
            rul_threshold: RUL threshold for anomaly

        Returns:
            Binary labels (1=anomaly, 0=normal)
        """
        labels = (test_df['RUL'] < rul_threshold).astype(int).values
        logger.info(f"Created labels using RUL threshold={rul_threshold}")
        logger.info(f"  Anomalies: {labels.sum()} / {len(labels)} ({100*labels.sum()/len(labels):.1f}%)")

        return labels

    def calculate_metrics(
        self,
        errors: np.ndarray,
        labels: np.ndarray,
        threshold: float
    ) -> Dict:
        """
        Calculate evaluation metrics.

        Args:
            errors: Reconstruction errors
            labels: True labels (1=anomaly, 0=normal)
            threshold: Anomaly threshold

        Returns:
            Dictionary with metrics
        """
        logger.info("Calculating evaluation metrics...")

        # Binary predictions
        predictions = (errors > threshold).astype(int)

        # =============================================================
        # TODO 2: Calculate precision, recall, and F1-score
        # =============================================================
        # HINT: Use precision_recall_fscore_support from sklearn.metrics
              
        #
        # Documentation: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
        #
        # YOUR CODE HERE (3 lines):
        precision, recall, f1, _ = precision_recall_fscore_support(
                  labels, predictions, average='binary'
                )


        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

        # ROC-AUC
        try:
            auc = roc_auc_score(labels, errors)
        except ValueError:
            auc = None
            logger.warning("Could not calculate AUC")

        metrics = {
            'threshold': float(threshold),
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'min_error': float(np.min(errors)),
            'max_error': float(np.max(errors)),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
            'auc': float(auc) if auc is not None else None
        }

        logger.info(f" Metrics calculated:")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall:    {recall:.3f}")
        logger.info(f"  F1-Score:  {f1:.3f}")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.3f}")

        return metrics

    def plot_error_distribution(
        self,
        errors: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        save_path: str = 'error_distribution.png'
    ):
        """Plot reconstruction error distribution."""
        plt.figure(figsize=(12, 5))

        # Subplot 1: Histogram
        plt.subplot(1, 2, 1)
        plt.hist(errors[labels == 0], bins=50, alpha=0.6, label='Normal', color='blue')
        plt.hist(errors[labels == 1], bins=50, alpha=0.6, label='Anomaly', color='red')
        plt.axvline(threshold, color='green', linestyle='--', linewidth=2, label='Threshold')
        plt.xlabel('Reconstruction Error', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Error Distribution', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        # Subplot 2: Box plot
        plt.subplot(1, 2, 2)
        data_to_plot = [errors[labels == 0], errors[labels == 1]]
        plt.boxplot(data_to_plot, tick_labels=['Normal', 'Anomaly'])
        plt.axhline(threshold, color='green', linestyle='--', linewidth=2, label='Threshold')
        plt.ylabel('Reconstruction Error', fontsize=12)
        plt.title('Error Box Plot', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        logger.info(f" Saved error distribution plot: {save_path}")

    def plot_roc_curve(
        self,
        labels: np.ndarray,
        errors: np.ndarray,
        save_path: str = 'roc_curve.png'
    ):
        """Plot ROC curve."""
        try:
            fpr, tpr, thresholds = roc_curve(labels, errors)
            auc = roc_auc_score(labels, errors)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('ROC Curve for Anomaly Detection', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()

            logger.info(f" Saved ROC curve: {save_path}")

        except Exception as e:
            logger.warning(f"Could not plot ROC curve: {e}")

    def plot_confusion_matrix(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        save_path: str = 'confusion_matrix.png'
    ):
        """Plot confusion matrix."""
        cm = confusion_matrix(labels, predictions)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'],
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

        logger.info(f" Saved confusion matrix: {save_path}")

    def evaluate(
        self,
        test_df: pd.DataFrame,
        rul_threshold: int = 30,
        threshold_method: str = 'percentile',
        threshold_percentile: float = 95.0,
        log_to_mlflow: bool = True
    ) -> Dict:
        """
        Complete evaluation pipeline.

        Args:
            test_df: Test DataFrame with RUL
            rul_threshold: RUL threshold for creating labels
            threshold_method: Method for determining anomaly threshold
            threshold_percentile: Percentile for threshold
            log_to_mlflow: Whether to log to MLflow

        Returns:
            Dictionary with evaluation results
        """
        print("\n" + "="*70)
        print("EVALUATING TURBOFAN AUTOENCODER")
        print("="*70 + "\n")

        # Calculate reconstruction errors
        errors = self.calculate_reconstruction_errors()

        # Determine threshold
        threshold = self.determine_threshold(
            errors,
            method=threshold_method,
            percentile=threshold_percentile
        )

        # Create labels from RUL
        labels = self.create_labels_from_rul(test_df, rul_threshold=rul_threshold)

        # Calculate metrics
        metrics = self.calculate_metrics(errors, labels, threshold)

        # Create visualizations
        self.plot_error_distribution(errors, labels, threshold)
        self.plot_roc_curve(labels, errors)

        predictions = (errors > threshold).astype(int)
        self.plot_confusion_matrix(labels, predictions)

        # Log to MLflow
        if log_to_mlflow:
            self._log_to_mlflow(metrics)

        print("\n" + "="*70)
        print("EVALUATION COMPLETE")
        print("="*70)
        print(f"Threshold: {threshold:.4f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall:    {metrics['recall']:.3f}")
        print(f"F1-Score:  {metrics['f1_score']:.3f}")
        print("="*70 + "\n")

        return metrics

    def _log_to_mlflow(self, metrics: Dict):
        """
        Log evaluation results to MLflow.

        Args:
            metrics: Dictionary with metrics
        """
        logger.info("Logging evaluation results to MLflow...")

        # =================================================================
        # TODO 3: Log metrics to MLflow
        # =================================================================
        # HINT: Use mlflow.log_metric(f"eval_{metric_name}", metric_value)
        #
        # Loop through all metrics in the dictionary and log them.
        # Add "eval_" prefix to distinguish from training metrics.
        #
        # Documentation: https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_metric
        #
        # YOUR CODE HERE (2-3 lines):
        # Example:
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                mlflow.log_metric(f"eval_{metric_name}", metric_value)




        # Log plots as artifacts
        mlflow.log_artifact('error_distribution.png')
        mlflow.log_artifact('roc_curve.png')
        mlflow.log_artifact('confusion_matrix.png')

        # Log metrics as JSON
        metrics_path = 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(metrics_path)
        os.remove(metrics_path)

        logger.info(" Logged evaluation results to MLflow")


def evaluate_model(
    model_uri: str,
    data_dir: str = 'data/processed/FD001',
    batch_size: int = 64,
    rul_threshold: int = 30,
    threshold_percentile: float = 95.0
) -> Dict:
    """
    Main evaluation function.

    This function handles all the setup. You don't need to modify this -
    focus on completing the TODOs above!

    Args:
        model_uri: MLflow model URI or local path
        data_dir: Directory with processed data
        batch_size: Batch size for evaluation
        rul_threshold: RUL threshold for creating labels
        threshold_percentile: Percentile for anomaly threshold

    Returns:
        Evaluation metrics dictionary
    """
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader, input_dim = prepare_data_for_training(
        data_dir=data_dir,
        task='autoencoder',
        batch_size=batch_size
    )

    # Load test DataFrame for RUL labels
    _, _, test_df = load_processed_data(data_dir)

    # Load model
    logger.info(f"Loading model from: {model_uri}")
    try:
        # Try loading from MLflow
        model = mlflow.pytorch.load_model(model_uri)
        logger.info(" Loaded model from MLflow")
    except Exception as e:
        logger.warning(f"Could not load from MLflow: {e}")
        logger.info("Trying to load as local PyTorch model...")

        # Load as local file
        model = TurbofanAutoencoder(input_dim=input_dim)
        model.load_state_dict(torch.load(model_uri, weights_only=False))
        logger.info(" Loaded model from local file")

    # Create evaluator
    evaluator = TurbofanEvaluator(
        model=model,
        test_loader=test_loader,
        device=device
    )

    # Evaluate
    results = evaluator.evaluate(
        test_df=test_df,
        rul_threshold=rul_threshold,
        threshold_percentile=threshold_percentile,
        log_to_mlflow=True
    )

    return results


def main():
    """
    Main execution with command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Evaluate turbofan autoencoder'
    )

    parser.add_argument(
        '--model_uri',
        type=str,
        required=True,
        help='MLflow model URI or local path'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed/FD001',
        help='Directory with processed data'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size'
    )
    parser.add_argument(
        '--rul_threshold',
        type=int,
        default=30,
        help='RUL threshold for anomaly labeling'
    )
    parser.add_argument(
        '--threshold_percentile',
        type=float,
        default=95.0,
        help='Percentile for anomaly threshold'
    )

    args = parser.parse_args()

    # Evaluate
    results = evaluate_model(
        model_uri=args.model_uri,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        rul_threshold=args.rul_threshold,
        threshold_percentile=args.threshold_percentile
    )


if __name__ == "__main__":
    main()
