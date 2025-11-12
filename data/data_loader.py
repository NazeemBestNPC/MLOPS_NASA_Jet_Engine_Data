"""
PyTorch DataLoader Utilities for Turbofan Data

This module provides DataLoader classes and utilities for loading
processed turbofan sensor data into PyTorch models.

Features:
- TensorDataset creation from pandas DataFrames
- Custom DataLoader with batching and shuffling
- Support for both regression (RUL) and autoencoder tasks
- Efficient data loading for training

Author: Feda Almuhisen
Course: M2  Processus Data
Institution: Aix-Marseille University
Year: 2025-2026
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, List
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TurbofanDataset(Dataset):
    """
    Custom PyTorch Dataset for turbofan sensor data.

    Supports both:
    - Regression: X → RUL prediction
    - Autoencoder: X → X reconstruction

    Attributes:
        features: Sensor feature tensor
        targets: Target tensor (RUL or features for autoencoder)
        task: Task type ('regression' or 'autoencoder')

    Example:
        >>> dataset = TurbofanDataset(df, task='autoencoder')
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch_x, batch_y in loader:
        ...     outputs = model(batch_x)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        target_col: str = 'RUL',
        task: str = 'autoencoder'
    ):
        """
        Initialize dataset.

        Args:
            df: DataFrame with sensor data
            feature_cols: List of feature columns (if None, auto-detect sensors)
            target_col: Name of target column (for regression)
            task: Task type ('regression' or 'autoencoder')
        """
        self.df = df
        self.task = task

        # Auto-detect sensor columns if not provided
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c.startswith('sensor_')]

        self.feature_cols = feature_cols
        self.target_col = target_col

        # Extract features
        self.features = torch.FloatTensor(df[feature_cols].values)

        # Extract targets based on task
        if task == 'autoencoder':
            # For autoencoder, target = input (reconstruction)
            self.targets = self.features.clone()
        elif task == 'regression':
            # For regression, target = RUL
            self.targets = torch.FloatTensor(df[target_col].values).unsqueeze(1)
        else:
            raise ValueError(f"Invalid task: {task}. Choose 'autoencoder' or 'regression'")

        logger.info(f"Created TurbofanDataset: {len(self)} samples, {len(feature_cols)} features, task={task}")

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get sample at index.

        Args:
            idx: Sample index

        Returns:
            Tuple of (features, target)
        """
        return self.features[idx], self.targets[idx]

    def get_feature_dim(self) -> int:
        """Return number of features."""
        return len(self.feature_cols)


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    task: str = 'autoencoder',
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        feature_cols: List of feature columns
        task: Task type ('autoencoder' or 'regression')
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader)

    Example:
        >>> train_loader, val_loader, test_loader = create_dataloaders(
        ...     train_df, val_df, test_df,
        ...     task='autoencoder',
        ...     batch_size=64
        ... )
    """
    logger.info(f"Creating DataLoaders (task={task}, batch_size={batch_size})...")

    # Create datasets
    train_dataset = TurbofanDataset(train_df, feature_cols=feature_cols, task=task)
    val_dataset = TurbofanDataset(val_df, feature_cols=feature_cols, task=task)
    test_dataset = TurbofanDataset(test_df, feature_cols=feature_cols, task=task)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    logger.info(f" Created DataLoaders:")
    logger.info(f"  Train: {len(train_loader)} batches ({len(train_dataset)} samples)")
    logger.info(f"  Val:   {len(val_loader)} batches ({len(val_dataset)} samples)")
    logger.info(f"  Test:  {len(test_loader)} batches ({len(test_dataset)} samples)")

    return train_loader, val_loader, test_loader


def load_processed_data(
    data_dir: str = 'data/processed/FD001'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load processed CSV files.

    Args:
        data_dir: Directory containing processed CSV files

    Returns:
        Tuple of (train_df, val_df, test_df)

    Example:
        >>> train_df, val_df, test_df = load_processed_data('data/processed/FD001')
    """
    data_path = Path(data_dir)

    logger.info(f"Loading processed data from {data_path}...")

    train_df = pd.read_csv(data_path / 'train_processed.csv')
    val_df = pd.read_csv(data_path / 'val_processed.csv')
    test_df = pd.read_csv(data_path / 'test_processed.csv')

    logger.info(f" Loaded data:")
    logger.info(f"  Train: {len(train_df):,} samples")
    logger.info(f"  Val:   {len(val_df):,} samples")
    logger.info(f"  Test:  {len(test_df):,} samples")

    return train_df, val_df, test_df.drop(columns=["sensor_10"])


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Extract sensor feature columns from DataFrame.

    Args:
        df: DataFrame to extract columns from

    Returns:
        List of sensor column names
    """

    return [c for c in df.columns if c.startswith('sensor_')]


def get_input_dim(df: pd.DataFrame) -> int:
    """
    Get input dimension (number of features).

    Args:
        df: DataFrame to get dimension from

    Returns:
        Number of sensor features
    """
    return len(get_feature_columns(df))


class EngineDataLoader:
    """
    Load data for specific engines.

    Useful for analyzing individual engine degradation patterns.

    Example:
        >>> loader = EngineDataLoader('data/processed/FD001')
        >>> engine_5_data = loader.get_engine_data(5)
        >>> loader.plot_engine_degradation(5)
    """

    def __init__(self, data_dir: str = 'data/processed/FD001'):
        """
        Initialize engine data loader.

        Args:
            data_dir: Directory with processed data
        """
        self.data_dir = Path(data_dir)
        self.train_df, self.val_df, self.test_df = load_processed_data(data_dir)
        self.all_df = pd.concat([self.train_df, self.val_df, self.test_df])

    def get_engine_data(self, engine_id: int) -> pd.DataFrame:
        """
        Get all data for specific engine.

        Args:
            engine_id: Engine ID to retrieve

        Returns:
            DataFrame with engine's complete history
        """
        return self.all_df[self.all_df['engine_id'] == engine_id].copy()

    def get_engine_ids(self, split: str = 'all') -> List[int]:
        """
        Get list of engine IDs.

        Args:
            split: Which split to get IDs from ('train', 'val', 'test', 'all')

        Returns:
            List of engine IDs
        """
        if split == 'train':
            return sorted(self.train_df['engine_id'].unique().tolist())
        elif split == 'val':
            return sorted(self.val_df['engine_id'].unique().tolist())
        elif split == 'test':
            return sorted(self.test_df['engine_id'].unique().tolist())
        else:
            return sorted(self.all_df['engine_id'].unique().tolist())

    def get_last_cycle_data(self, engine_id: int) -> pd.Series:
        """
        Get sensor values at last observed cycle for an engine.

        Args:
            engine_id: Engine ID

        Returns:
            Series with last cycle sensor values
        """
        engine_data = self.get_engine_data(engine_id)
        return engine_data.iloc[-1]

    def plot_engine_degradation(self, engine_id: int, sensors_to_plot: Optional[List[str]] = None):
        """
        Plot sensor degradation over time for an engine.

        Args:
            engine_id: Engine ID to plot
            sensors_to_plot: Specific sensors to plot (if None, plot first 4)
        """
        import matplotlib.pyplot as plt

        engine_data = self.get_engine_data(engine_id)

        if sensors_to_plot is None:
            sensors_to_plot = get_feature_columns(engine_data)[:4]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for idx, sensor in enumerate(sensors_to_plot):
            ax = axes[idx]
            ax.plot(engine_data['cycle'], engine_data[sensor])
            ax.set_xlabel('Cycle')
            ax.set_ylabel(sensor)
            ax.set_title(f'Engine {engine_id} - {sensor}')
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(f'engine_{engine_id}_degradation.png')
        logger.info(f" Saved plot: engine_{engine_id}_degradation.png")
        plt.close()


def prepare_data_for_training(
    data_dir: str = 'data/processed/FD001',
    task: str = 'autoencoder',
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    One-stop function to prepare all data for training.

    This is the main function you'll use in train.py.

    Args:
        data_dir: Directory with processed data
        task: Task type ('autoencoder' or 'regression')
        batch_size: Batch size

    Returns:
        Tuple of (train_loader, val_loader, test_loader, input_dim)

    Example:
        >>> train_loader, val_loader, test_loader, input_dim = prepare_data_for_training()
        >>> model = Autoencoder(input_dim=input_dim)
        >>> for batch_x, batch_y in train_loader:
        ...     outputs = model(batch_x)
    """
    logger.info("\n" + "="*70)
    logger.info("PREPARING DATA FOR TRAINING")
    logger.info("="*70)

    # Load processed data
    train_df, val_df, test_df = load_processed_data(data_dir)

    # Get input dimension
    input_dim = get_input_dim(train_df)
    logger.info(f"Input dimension: {input_dim} features")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        task=task,
        batch_size=batch_size
    )

    logger.info("="*70 + "\n")

    return train_loader, val_loader, test_loader, input_dim


def main():
    """
    Demonstration of data loading utilities.
    """
    # Assume data has been processed
    data_dir = 'data/processed/FD001'

    # Method 1: Simple one-function approach
    print("\n### Method 1: Simple approach ###")
    train_loader, val_loader, test_loader, input_dim = prepare_data_for_training(
        data_dir=data_dir,
        task='autoencoder',
        batch_size=64
    )

    print(f"\nInput dimension: {input_dim}")
    print(f"Train batches: {len(train_loader)}")

    # Test loading a batch
    batch_x, batch_y = next(iter(train_loader))
    print(f"\nBatch shape: {batch_x.shape}")
    print(f"Target shape: {batch_y.shape}")

    # Method 2: Load and explore specific engines
    print("\n### Method 2: Engine-specific analysis ###")
    engine_loader = EngineDataLoader(data_dir)

    train_engines = engine_loader.get_engine_ids('train')
    print(f"\nTrain engines: {len(train_engines)}")
    print(f"First 5 engine IDs: {train_engines[:5]}")

    # Get data for engine 1
    engine_1 = engine_loader.get_engine_data(1)
    print(f"\nEngine 1 data shape: {engine_1.shape}")
    print(f"Engine 1 cycles: {engine_1['cycle'].min()} to {engine_1['cycle'].max()}")

    # Plot engine degradation (if matplotlib available)
    try:
        engine_loader.plot_engine_degradation(1)
    except ImportError:
        print("Matplotlib not available for plotting")


if __name__ == "__main__":
    main()
