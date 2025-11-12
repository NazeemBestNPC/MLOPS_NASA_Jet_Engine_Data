#!/usr/bin/env python3
"""
Turbofan Autoencoder Training Script with MLflow Tracking

STUDENTS: You will complete 5 TODOs to integrate MLflow tracking.

What you'll learn:
- Setting up MLflow experiments
- Logging hyperparameters
- Logging metrics per epoch
- Saving models to MLflow

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
from typing import Tuple, Dict
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# MLflow imports
import mlflow
import mlflow.pytorch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.autoencoder import TurbofanAutoencoder, print_model_summary
from data.data_loader import prepare_data_for_training

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TurbofanTrainer:
    """
    Trainer class for turbofan autoencoder with MLflow tracking.

    This class handles the training loop and integrates with MLflow
    to track all experiments automatically.
    """

    def __init__(
        self,
        model: TurbofanAutoencoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cpu'
    ):
        """
        Initialize trainer.

        Args:
            model: TurbofanAutoencoder instance
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # History for plotting
        self.train_losses = []
        self.val_losses = []

        logger.info(f"Initialized trainer on device: {device}")

    def train_epoch(
        self,
        optimizer: optim.Optimizer,
        criterion: nn.Module
    ) -> float:
        """
        Train for one epoch.

        Args:
            optimizer: Optimizer
            criterion: Loss function

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0
        num_batches = 0

        # Progress bar
        pbar = tqdm(self.train_loader, desc='Training', leave=False)

        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass
            outputs = self.model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, criterion: nn.Module) -> float:
        """
        Validate model.

        Args:
            criterion: Loss function

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(
        self,
        epochs: int = 50,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        experiment_name: str = "turbofan_autoencoder",
        run_name: str = None,
        early_stopping_patience: int = 10
    ) -> Dict:
        """
        Complete training loop with MLflow tracking.

        STUDENTS: Complete the 5 TODOs below to integrate MLflow!

        Args:
            epochs: Number of epochs
            lr: Learning rate
            weight_decay: Weight decay for regularization
            experiment_name: MLflow experiment name
            run_name: MLflow run name (optional)
            early_stopping_patience: Epochs to wait before early stopping

        Returns:
            Dictionary with training results
        """
        print("\n" + "="*70)
        print("TRAINING TURBOFAN AUTOENCODER")
        print("="*70 + "\n")

        # =================================================================
        # TODO 1: Set MLflow experiment name
        # =================================================================
        # HINT: Use mlflow.set_experiment(experiment_name)
        #
        # This creates or selects an experiment where all your runs
        # will be grouped together.
        #
        # Documentation: https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_experiment
        #
        # YOUR CODE HERE (1 line):

        mlflow.set_experiment(experiment_name)
        logger.info(f" MLflow experiment: {experiment_name}")

        # =================================================================
        # TODO 2: Start MLflow run
        # =================================================================
        # HINT: Use 'with mlflow.start_run(run_name=run_name) as run:'
        #
        # IMPORTANT: All MLflow logging (TODOs 3-5) must happen INSIDE
        # this with block! Make sure to indent the code below.
        #
        # Documentation: https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run
        #
        # YOUR CODE HERE (1 line - then indent everything below):
        with mlflow.start_run(run_name=run_name) as run:

            logger.info(f" MLflow run started")

            # =============================================================
            # TODO 3: Log hyperparameters
            # =============================================================
            # HINT: Use mlflow.log_param("parameter_name", value)
            #
            # Log these hyperparameters:
            # - epochs
            # - learning_rate (use 'lr' variable)
            # - weight_decay
            # - batch_size (get from self.train_loader.batch_size)
            # - encoding_dim (get from self.model.encoding_dim)
            #
            # Documentation: https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_param
            #
            # YOUR CODE HERE (5 lines):


            mlflow.log_param("epochs", epochs)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("weight_decay", weight_decay)
            mlflow.log_param("batch_size", self.train_loader.batch_size)
            mlflow.log_param("encoding_dim", self.model.encoding_dim)

            logger.info(f" Logged hyperparameters")

            # Setup training
            criterion = nn.MSELoss()
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

            # Early stopping variables
            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None

            # Training loop
            print(f"Starting training for {epochs} epochs...\n")

            for epoch in range(epochs):
                # Train
                train_loss = self.train_epoch(optimizer, criterion)
                self.train_losses.append(train_loss)

                # Validate
                val_loss = self.validate(criterion)
                self.val_losses.append(val_loss)

                # ==========================================================
                # TODO 4: Log metrics per epoch
                # ==========================================================
                # HINT: Use mlflow.log_metric("metric_name", value, step=epoch)
                #
                # Log these two metrics:
                # - 'train_loss' with value train_loss
                # - 'val_loss' with value val_loss
                #
                # The step parameter allows MLflow to show the curve over time!
                #
                # Documentation: https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.log_metric
                #
                # YOUR CODE HERE (2 lines):

                mlflow.log_metrics(metrics = {"train_loss": train_loss, "val_loss": val_loss})
                # Print progress
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f}")

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                    logger.info(f"  → New best validation loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break

            # Restore best model
            if best_model_state is not None:
                self.model.load_state_dict(best_model_state)
                logger.info(" Restored best model weights")

            # Log final metrics
            mlflow.log_metric('final_train_loss', self.train_losses[-1])
            mlflow.log_metric('final_val_loss', self.val_losses[-1])
            mlflow.log_metric('best_val_loss', best_val_loss)
            mlflow.log_metric('epochs_trained', len(self.train_losses))

            # Plot and log training curves
            self._plot_training_curves()
            mlflow.log_artifact('training_curves.png')
            logger.info(" Logged training curves")

            # =============================================================
            # TODO 5: Log the trained model
            # =============================================================
            # HINT: Use mlflow.pytorch.log_model(self.model, "model")
            #
            # This saves the entire PyTorch model to MLflow so you can:
            # - Load it later for evaluation
            # - Deploy it to production
            # - Compare different model versions
            #
            # Documentation: https://www.mlflow.org/docs/latest/python_api/mlflow.pytorch.html#mlflow.pytorch.log_model
            #
            # YOUR CODE HERE (1 line):

            mlflow.pytorch.log_model(self.model, "model")

            logger.info(" Logged model to MLflow")

            # Create summary
            summary = {
                'final_train_loss': float(self.train_losses[-1]),
                'final_val_loss': float(self.val_losses[-1]),
                'best_val_loss': float(best_val_loss),
                'epochs_trained': len(self.train_losses)
            }

            print("\n" + "="*70)
            print("TRAINING COMPLETE")
            print("="*70)
            print(f"Best validation loss: {best_val_loss:.4f}")
            print(f"Epochs trained: {len(self.train_losses)}")
            print("="*70 + "\n")

            return summary

    def _plot_training_curves(self):
        """Plot and save training curves."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title('Training and Validation Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150)
        plt.close()


def train_model(
    data_dir: str = 'data/processed/FD001',
    epochs: int = 50,
    lr: float = 0.001,
    batch_size: int = 64,
    encoding_dim: int = 4,
    hidden_dim1: int = 16,
    hidden_dim2: int = 8,
    dropout_rate: float = 0.1,
    experiment_name: str = "other_turbofan_autoencoder_2",
    run_name: str = None
) -> Dict:
    """
    Main training function (high-level API).

    This function handles all the setup and calls the trainer.
    You don't need to modify this function - focus on the TODOs above!

    Args:
        data_dir: Directory with processed data
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        encoding_dim: Bottleneck dimension
        hidden_dim1: First hidden layer size
        hidden_dim2: Second hidden layer size
        dropout_rate: Dropout rate
        experiment_name: MLflow experiment name
        run_name: MLflow run name

    Returns:
        Training summary dictionary
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

    # Create model
    model = TurbofanAutoencoder(
        input_dim=input_dim,
        encoding_dim=encoding_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        dropout_rate=dropout_rate
    )

    print_model_summary(model)

    # Create trainer
    trainer = TurbofanTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    # Train
    results = trainer.train(
        epochs=epochs,
        lr=lr,
        experiment_name=experiment_name,
        run_name=run_name
    )

    return results


def main():
    """
    Main execution with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description='Train turbofan autoencoder with MLflow tracking'
    )

    # Data arguments
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/processed/FD001',
        help='Directory with processed data'
    )

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    # Model arguments
    parser.add_argument('--encoding_dim', type=int, default=4, help='Encoding dimension')
    parser.add_argument('--hidden_dim1', type=int, default=16, help='Hidden dimension 1')
    parser.add_argument('--hidden_dim2', type=int, default=8, help='Hidden dimension 2')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')

    # MLflow arguments
    parser.add_argument(
        '--experiment_name',
        type=str,
        default='turbofan_autoencoder',
        help='MLflow experiment name'
    )
    parser.add_argument('--run_name', type=str, default=None, help='MLflow run name')

    args = parser.parse_args()

    # Train
    results = train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        encoding_dim=args.encoding_dim,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        dropout_rate=args.dropout_rate,
        experiment_name=args.experiment_name,
        run_name=args.run_name
    )

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Open a new terminal")
    print("2. Activate your virtual environment")
    print("3. Run: mlflow ui")
    print("4. Open browser: http://localhost:5000")
    print("5. Explore your experiments!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
