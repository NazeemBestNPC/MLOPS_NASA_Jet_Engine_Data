#!/usr/bin/env python3
"""
NASA C-MAPSS Turbofan Data Preprocessing Pipeline

This module handles complete preprocessing of raw turbofan sensor data:
- RUL (Remaining Useful Life) calculation
- Sensor feature engineering
- Normalization and scaling
- Train/validation splits
- Data quality checks

Key Transformations:
1. Calculate RUL for each cycle (max_cycle - current_cycle)
2. Remove constant sensors (zero variance)
3. Normalize sensor values (StandardScaler)
4. Handle operational settings
5. Create time windows for sequence models (optional)

Author: Feda Almuhisen
Course: M2 SID - Processus Data
Institution: Aix-Marseille University
Year: 2025-2026
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TurbofanPreprocessor:
    """
    Complete preprocessing pipeline for NASA C-MAPSS turbofan data.

    This class handles all data transformations needed to prepare
    raw sensor data for machine learning models.

    Attributes:
        scaler: Fitted StandardScaler for sensor normalization
        feature_cols: List of features to use for modeling
        removed_sensors: Sensors removed due to zero variance

    Example:
        >>> preprocessor = TurbofanPreprocessor()
        >>> train_df = preprocessor.add_rul(train_df)
        >>> train_scaled = preprocessor.fit_transform(train_df)
        >>> preprocessor.save_scaler('scaler.pkl')
    """

    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize preprocessor.

        Args:
            scaler_type: Type of scaler ('standard' or 'minmax')
        """
        self.scaler_type = scaler_type
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Invalid scaler_type: {scaler_type}")

        self.feature_cols: List[str] = []
        self.removed_sensors: List[str] = []
        self.is_fitted: bool = False

        logger.info(f"Initialized TurbofanPreprocessor with {scaler_type} scaler")

    def add_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Remaining Useful Life (RUL) for each engine cycle.

        RUL = max_cycle - current_cycle for each engine

        Args:
            df: DataFrame with 'engine_id' and 'cycle' columns

        Returns:
            DataFrame with added 'RUL' column

        Example:
            Engine 1: cycles 1, 2, 3, ..., 100
            RUL:      99, 98, 97, ..., 0
        """
        logger.info("Calculating RUL (Remaining Useful Life)...")

        # Make a copy to avoid modifying original
        df = df.copy()

        # Calculate max cycle for each engine
        max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
        max_cycles.columns = ['engine_id', 'max_cycle']

        # Merge and calculate RUL
        df = df.merge(max_cycles, on='engine_id', how='left')
        df['RUL'] = df['max_cycle'] - df['cycle']
        df = df.drop(columns=['max_cycle'])

        logger.info(f" RUL calculated. Range: {df['RUL'].min()} to {df['RUL'].max()}")

        return df

    def add_rul_from_test(
        self,
        test_df: pd.DataFrame,
        rul_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add RUL to test set using provided RUL values.

        Test set RUL is more complex:
        - We know the TRUE RUL at the last observed cycle
        - Need to calculate RUL for all previous cycles

        Args:
            test_df: Test DataFrame
            rul_df: DataFrame with true RUL values (engine_id, RUL)

        Returns:
            Test DataFrame with RUL column
        """
        logger.info("Adding RUL to test set...")

        test_df = test_df.copy()

        # Get last cycle for each engine
        last_cycles = test_df.groupby('engine_id')['cycle'].max().reset_index()
        last_cycles.columns = ['engine_id', 'last_cycle']

        # Merge with true RUL at last cycle
        test_df = test_df.merge(last_cycles, on='engine_id', how='left')
        test_df = test_df.merge(rul_df, on='engine_id', how='left')

        # Calculate RUL for each cycle
        # RUL at cycle = RUL_at_last + (last_cycle - current_cycle)
        test_df['RUL'] = test_df['RUL'] + (test_df['last_cycle'] - test_df['cycle'])
        test_df = test_df.drop(columns=['last_cycle'])

        logger.info(f" Test RUL calculated. Range: {test_df['RUL'].min()} to {test_df['RUL'].max()}")

        return test_df

    def remove_constant_features(
        self,
        df: pd.DataFrame,
        variance_threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Remove features with zero or near-zero variance.

        Constant sensors provide no information for learning.

        Args:
            df: DataFrame with sensor columns
            variance_threshold: Minimum variance to keep feature

        Returns:
            DataFrame with constant features removed
        """
        logger.info("Identifying constant features...")

        # Identify sensor columns
        sensor_cols = [c for c in df.columns if c.startswith('sensor_')]

        # Calculate variance for each sensor
        variances = df[sensor_cols].var()

        # Find constant sensors
        constant_sensors = variances[variances <= variance_threshold].index.tolist()

        if constant_sensors:
            logger.info(f"Removing {len(constant_sensors)} constant sensors: {constant_sensors}")
            df = df.drop(columns=constant_sensors)
            self.removed_sensors.extend(constant_sensors)
        else:
            logger.info(" No constant sensors found")

        return df

    def remove_settings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optionally remove operational setting columns.

        Operational settings may or may not be useful depending on the dataset.
        FD001: Single condition, settings not useful
        FD002/FD004: Multiple conditions, settings ARE useful

        Args:
            df: DataFrame with setting columns

        Returns:
            DataFrame without setting columns
        """
        setting_cols = [c for c in df.columns if c.startswith('setting_')]

        if setting_cols:
            logger.info(f"Removing operational settings: {setting_cols}")
            df = df.drop(columns=setting_cols)

        return df

    def get_sensor_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of all sensor columns in DataFrame.

        Args:
            df: DataFrame to extract sensor columns from

        Returns:
            List of sensor column names
        """
        return [c for c in df.columns if c.startswith('sensor_')]

    def fit_transform(
        self,
        df: pd.DataFrame,
        exclude_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fit scaler and transform sensor features.

        This should ONLY be called on training data.

        Args:
            df: Training DataFrame
            exclude_cols: Columns to exclude from scaling (e.g., 'RUL')

        Returns:
            DataFrame with scaled sensor features
        """
        logger.info("Fitting scaler and transforming features...")

        df = df.copy()

        # Default exclude columns
        if exclude_cols is None:
            exclude_cols = ['engine_id', 'cycle', 'RUL']

        # Get sensor columns to scale
        sensor_cols = self.get_sensor_columns(df)
        self.feature_cols = sensor_cols

        # Fit and transform
        df[sensor_cols] = self.scaler.fit_transform(df[sensor_cols])
        self.is_fitted = True

        logger.info(f" Fitted scaler on {len(sensor_cols)} features")
        logger.info(f"  Feature range: [{df[sensor_cols].min().min():.3f}, {df[sensor_cols].max().max():.3f}]")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using already fitted scaler.

        Use this for validation/test data.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit_transform() first.")

        logger.info("Transforming features with fitted scaler...")

        df = df.copy()

        # Transform using fitted scaler
        df[self.feature_cols] = self.scaler.transform(df[self.feature_cols])

        logger.info(f" Transformed {len(self.feature_cols)} features")

        return df

    def clip_rul(
        self,
        df: pd.DataFrame,
        max_rul: int = 125
    ) -> pd.DataFrame:
        """
        Clip RUL values to maximum threshold.

        Rationale: Early in engine life, exact RUL doesn't matter much.
        Only care about critical period near failure.

        Args:
            df: DataFrame with RUL column
            max_rul: Maximum RUL value to keep

        Returns:
            DataFrame with clipped RUL
        """
        logger.info(f"Clipping RUL to max value of {max_rul}...")

        df = df.copy()
        original_max = df['RUL'].max()

        df['RUL'] = df['RUL'].clip(upper=max_rul)

        logger.info(f" RUL clipped: {original_max} → {df['RUL'].max()}")

        return df

    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int = 30,
        sequence_stride: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM/RNN models.

        Args:
            df: DataFrame with sensor data
            sequence_length: Length of each sequence (time steps)
            sequence_stride: Stride between sequences

        Returns:
            Tuple of (X_sequences, y_rul)

        Example:
            sequence_length=30 means last 30 cycles predict RUL
        """
        logger.info(f"Creating sequences (length={sequence_length}, stride={sequence_stride})...")

        sequences = []
        targets = []

        # Group by engine
        for engine_id, engine_df in df.groupby('engine_id'):
            engine_df = engine_df.sort_values('cycle')

            # Get sensor values and RUL
            sensor_values = engine_df[self.feature_cols].values
            rul_values = engine_df['RUL'].values

            # Create sliding windows
            for i in range(0, len(sensor_values) - sequence_length + 1, sequence_stride):
                seq = sensor_values[i:i + sequence_length]
                target = rul_values[i + sequence_length - 1]

                sequences.append(seq)
                targets.append(target)

        sequences = np.array(sequences)
        targets = np.array(targets)

        logger.info(f" Created {len(sequences)} sequences of shape {sequences.shape}")

        return sequences, targets

    def split_train_val(
        self,
        df: pd.DataFrame,
        val_size: float = 0.2,
        random_state: int = 42,
        split_by_engine: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and validation sets.

        Args:
            df: DataFrame to split
            val_size: Fraction for validation
            random_state: Random seed
            split_by_engine: If True, split engines (better validation)

        Returns:
            Tuple of (train_df, val_df)
        """
        logger.info(f"Splitting data (val_size={val_size}, split_by_engine={split_by_engine})...")

        if split_by_engine:
            # Split by engines (recommended)
            engine_ids = df['engine_id'].unique()
            train_engines, val_engines = train_test_split(
                engine_ids,
                test_size=val_size,
                random_state=random_state
            )

            train_df = df[df['engine_id'].isin(train_engines)]
            val_df = df[df['engine_id'].isin(val_engines)]

            logger.info(f" Split: {len(train_engines)} train engines, {len(val_engines)} val engines")

        else:
            # Random split (less realistic)
            train_df, val_df = train_test_split(
                df,
                test_size=val_size,
                random_state=random_state
            )

            logger.info(f" Split: {len(train_df)} train samples, {len(val_df)} val samples")

        return train_df, val_df

    def get_preprocessing_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate preprocessing statistics report.

        Args:
            df: Preprocessed DataFrame

        Returns:
            Dictionary with preprocessing statistics
        """
        sensor_cols = self.get_sensor_columns(df)

        report = {
            'total_samples': len(df),
            'num_engines': df['engine_id'].nunique(),
            'num_features': len(sensor_cols),
            'removed_features': len(self.removed_sensors),
            'removed_feature_names': self.removed_sensors,
            'rul_min': float(df['RUL'].min()),
            'rul_max': float(df['RUL'].max()),
            'rul_mean': float(df['RUL'].mean()),
            'cycles_per_engine_avg': float(df.groupby('engine_id')['cycle'].max().mean()),
            'scaler_type': self.scaler_type,
            'feature_columns': sensor_cols,
        }

        return report

    def save_preprocessor(self, save_dir: str = 'data/processed'):
        """
        Save fitted preprocessor (scaler + metadata).

        Args:
            save_dir: Directory to save preprocessor
        """
        import pickle

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save scaler
        scaler_path = save_path / 'scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        # Save metadata
        metadata = {
            'scaler_type': self.scaler_type,
            'feature_cols': self.feature_cols,
            'removed_sensors': self.removed_sensors,
            'is_fitted': self.is_fitted
        }

        metadata_path = save_path / 'preprocessor_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f" Saved preprocessor to {save_path}")

    @classmethod
    def load_preprocessor(cls, load_dir: str = 'data/processed') -> 'TurbofanPreprocessor':
        """
        Load saved preprocessor.

        Args:
            load_dir: Directory to load from

        Returns:
            Loaded TurbofanPreprocessor instance
        """
        import pickle

        load_path = Path(load_dir)

        # Load metadata
        metadata_path = load_path / 'preprocessor_metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Create instance
        preprocessor = cls(scaler_type=metadata['scaler_type'])

        # Load scaler
        scaler_path = load_path / 'scaler.pkl'
        with open(scaler_path, 'rb') as f:
            preprocessor.scaler = pickle.load(f)

        # Restore metadata
        preprocessor.feature_cols = metadata['feature_cols']
        preprocessor.removed_sensors = metadata['removed_sensors']
        preprocessor.is_fitted = metadata['is_fitted']

        logger.info(f" Loaded preprocessor from {load_path}")

        return preprocessor


def preprocess_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    clip_rul: bool = True,
    max_rul: int = 125,
    remove_settings: bool = True,
    save_dir: str = 'data/processed'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TurbofanPreprocessor]:
    """
    Complete preprocessing pipeline (one function to rule them all).

    Args:
        train_df: Training data
        test_df: Test data
        rul_df: Test RUL values
        clip_rul: Whether to clip RUL values
        max_rul: Maximum RUL threshold
        remove_settings: Whether to remove operational settings
        save_dir: Directory to save processed data

    Returns:
        Tuple of (train_processed, val_processed, test_processed, preprocessor)
    """
    print("\n" + "="*70)
    print("TURBOFAN DATA PREPROCESSING PIPELINE")
    print("="*70 + "\n")

    # Initialize preprocessor
    preprocessor = TurbofanPreprocessor(scaler_type='standard')

    # 1. Add RUL to training data
    train_df = preprocessor.add_rul(train_df)

    # 2. Add RUL to test data
    test_df = preprocessor.add_rul_from_test(test_df, rul_df)

    # 3. Remove constant sensors
    train_df = preprocessor.remove_constant_features(train_df)
    test_df = preprocessor.remove_constant_features(test_df)

    # 4. Optionally remove settings (for FD001)
    if remove_settings:
        train_df = preprocessor.remove_settings(train_df)
        test_df = preprocessor.remove_settings(test_df)

    # 5. Clip RUL
    if clip_rul:
        train_df = preprocessor.clip_rul(train_df, max_rul=max_rul)
        test_df = preprocessor.clip_rul(test_df, max_rul=max_rul)

    # 6. Split train into train/val
    train_df, val_df = preprocessor.split_train_val(
        train_df,
        val_size=0.2,
        split_by_engine=True
    )

    # 7. Fit scaler on training data and transform all sets
    train_df = preprocessor.fit_transform(train_df)
    val_df = preprocessor.transform(val_df)
    test_df = preprocessor.transform(test_df)

    # 8. Save processed data
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(save_path / 'train_processed.csv', index=False)
    val_df.to_csv(save_path / 'val_processed.csv', index=False)
    test_df.to_csv(save_path / 'test_processed.csv', index=False)

    logger.info(f" Saved processed data to {save_path}")

    # 9. Save preprocessor
    preprocessor.save_preprocessor(save_dir)

    # 10. Generate and save report
    report = preprocessor.get_preprocessing_report(train_df)
    report_path = save_path / 'preprocessing_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    logger.info(f" Saved preprocessing report to {report_path}")

    # Print summary
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"Training samples: {len(train_df):,}")
    print(f"Validation samples: {len(val_df):,}")
    print(f"Test samples: {len(test_df):,}")
    print(f"Features: {len(preprocessor.feature_cols)}")
    print(f"Removed features: {len(preprocessor.removed_sensors)}")
    print("="*70 + "\n")

    return train_df, val_df, test_df, preprocessor


def main():
    """
    Main execution function demonstrating preprocessing workflow.
    """
    from download_data import TurbofanDataDownloader

    # Load raw data
    print("Loading raw data...")
    downloader = TurbofanDataDownloader()
    train_df, test_df, rul_df = downloader.load_dataset('FD001')

    # Run preprocessing pipeline
    train_processed, val_processed, test_processed, preprocessor = preprocess_pipeline(
        train_df=train_df,
        test_df=test_df,
        rul_df=rul_df,
        clip_rul=True,
        max_rul=125,
        remove_settings=True,
        save_dir='data/processed/FD001'
    )

    # Display sample
    print("\nProcessed training data (first 5 rows):")
    print(train_processed.head())

    print("\nFeature statistics:")
    print(train_processed[preprocessor.feature_cols].describe())


if __name__ == "__main__":
    main()
