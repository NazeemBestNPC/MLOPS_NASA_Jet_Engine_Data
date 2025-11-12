
"""
NASA C-MAPSS Turbofan Engine Degradation Dataset Downloader

This script downloads the NASA Commercial Modular Aero-Propulsion System
Simulation (C-MAPSS) dataset for predictive maintenance research.

Dataset Details:
- 4 sub-datasets (FD001, FD002, FD003, FD004)
- 21 sensor measurements
- 3 operational settings
- Target: Remaining Useful Life (RUL) prediction

Source: NASA Ames Prognostics Data Repository
Alternative: Kaggle mirror for reliability

Author: Feda Almuhisen
Course: M2 SID - Processus Data
Institution: Aix-Marseille University
Year: 2025-2026
"""

import os
import zipfile
import logging
from pathlib import Path
from typing import Tuple, List
import pandas as pd
import numpy as np
from urllib.request import urlretrieve
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TurbofanDataDownloader:
    """
    Download and extract NASA C-MAPSS Turbofan dataset.

    The dataset contains run-to-failure sensor data from turbofan engines
    under different operating conditions and fault modes.

    Attributes:
        data_dir: Directory to save downloaded data
        raw_dir: Directory for raw unprocessed data
        url: URL to download dataset from

    Example:
        >>> downloader = TurbofanDataDownloader()
        >>> downloader.download_all()
        >>> train_df, test_df, rul_df = downloader.load_dataset('FD001')
    """

    # Kaggle mirror (more reliable than NASA direct download)
    KAGGLE_URL = "https://ti.arc.nasa.gov/c/6/"
    NASA_URL = "https://ti.arc.nasa.gov/c/6/"

    # Alternative: Direct file URLs (if above fails)
    BACKUP_URL = "https://github.com/hfldai/DCNN/raw/master/data/train_FD001.txt"

    def __init__(self, data_dir: str = "data"):
        """
        Initialize downloader.

        Args:
            data_dir: Base directory for data storage (default: "data")
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Column names for the dataset
        self.index_cols = ['engine_id', 'cycle']
        self.setting_cols = ['setting_1', 'setting_2', 'setting_3']
        self.sensor_cols = [f'sensor_{i}' for i in range(1, 22)]  # 21 sensors
        self.all_cols = self.index_cols + self.setting_cols + self.sensor_cols

        logger.info(f"Initialized downloader. Data directory: {self.raw_dir.absolute()}")

    def download_from_url(self, url: str, output_path: Path) -> bool:
        """
        Download file from URL with progress bar.

        Args:
            url: URL to download from
            output_path: Path to save file

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading from {url}...")

            # Custom progress bar hook
            def reporthook(block_num, block_size, total_size):
                if hasattr(reporthook, 'pbar'):
                    reporthook.pbar.update(block_size)
                else:
                    reporthook.pbar = tqdm(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        desc='Downloading'
                    )

            urlretrieve(url, output_path, reporthook)

            if hasattr(reporthook, 'pbar'):
                reporthook.pbar.close()

            logger.info(f" Downloaded to {output_path}")
            return True

        except Exception as e:
            logger.error(f" Download failed: {e}")
            return False

    def download_dataset_manual(self) -> bool:
        """
        Provide manual download instructions.

        The NASA dataset requires manual download from Kaggle.
        This method provides clear instructions to the user.

        Returns:
            True if user confirms download, False otherwise
        """
        print("\n" + "="*70)
        print("MANUAL DOWNLOAD REQUIRED")
        print("="*70)
        print("\nThe NASA C-MAPSS dataset requires manual download from Kaggle.")
        print("\n DOWNLOAD INSTRUCTIONS:")
        print("\n1. Visit Kaggle:")
        print("   https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
        print("\n2. Click 'Download' button")
        print("   (You may need to create a free Kaggle account)")
        print("\n3. This downloads 'archive.zip' to your Downloads folder")
        print(f"\n4. Move it to: {self.raw_dir.absolute()}")
        print("   and rename to: CMAPSSData.zip")
        print("\n   Command:")
        print(f"   mv ~/Downloads/archive.zip {self.raw_dir.absolute()}/CMAPSSData.zip")
        print("\n5. Run this script again to extract:")
        print("   python data/download_data.py")
        print("\n" + "="*70)
        print("ALTERNATIVE - Using Kaggle API (Advanced):")
        print("="*70)
        print("\n1. Install: pip install kaggle")
        print("2. Get API key from: https://www.kaggle.com/settings")
        print("3. Place kaggle.json in ~/.kaggle/")
        print("4. Run: kaggle datasets download -d behrad3d/nasa-cmaps -p data/raw --unzip")
        print("="*70)

        # Check if ZIP already exists
        zip_path = self.raw_dir / "CMAPSSData.zip"
        if zip_path.exists():
            logger.info(f" Found ZIP file: {zip_path}")
            return True
        else:
            logger.warning(" ZIP file not found. Please download manually.")
            return False

    def extract_zip(self, zip_path: Path) -> bool:
        """
        Extract downloaded ZIP file.

        Args:
            zip_path: Path to ZIP file

        Returns:
            True if extraction successful, False otherwise
        """
        try:
            logger.info(f"Extracting {zip_path.name}...")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract all files
                zip_ref.extractall(self.raw_dir)

            logger.info(" Extraction complete")
            return True

        except Exception as e:
            logger.error(f" Extraction failed: {e}")
            return False

    def load_raw_file(self, filename: str) -> pd.DataFrame:
        """
        Load raw text file into pandas DataFrame.

        The C-MAPSS data files are space-separated text files without headers.

        Args:
            filename: Name of file (e.g., 'train_FD001.txt')

        Returns:
            Loaded DataFrame with proper column names
        """
        filepath = self.raw_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        logger.info(f"Loading {filename}...")

        # Load space-separated file
        df = pd.read_csv(
            filepath,
            sep=r'\s+',  # Multiple spaces
            header=None,
            names=self.all_cols
        )

        logger.info(f" Loaded {len(df):,} samples")
        return df

    def load_rul_file(self, filename: str) -> pd.DataFrame:
        """
        Load RUL (Remaining Useful Life) file for test set.

        Test set RUL files contain one value per engine (true RUL at end).

        Args:
            filename: Name of RUL file (e.g., 'RUL_FD001.txt')

        Returns:
            DataFrame with engine_id and true RUL
        """
        filepath = self.raw_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        logger.info(f"Loading {filename}...")

        # Load single column file
        rul_values = pd.read_csv(filepath, header=None, names=['RUL'])
        rul_values['engine_id'] = range(1, len(rul_values) + 1)

        logger.info(f" Loaded RUL for {len(rul_values)} engines")
        return rul_values[['engine_id', 'RUL']]

    def load_dataset(self, dataset_name: str = 'FD001') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load complete dataset (train, test, RUL).

        Args:
            dataset_name: Dataset to load ('FD001', 'FD002', 'FD003', or 'FD004')

        Returns:
            Tuple of (train_df, test_df, rul_df)

        Example:
            >>> train, test, rul = downloader.load_dataset('FD001')
            >>> print(f"Train: {len(train)} samples, Test: {len(test)} samples")
        """
        valid_datasets = ['FD001', 'FD002', 'FD003', 'FD004']
        if dataset_name not in valid_datasets:
            raise ValueError(f"Invalid dataset. Choose from {valid_datasets}")

        logger.info(f"\nLoading {dataset_name} dataset...")

        # Load files
        train_df = self.load_raw_file(f'train_{dataset_name}.txt')
        test_df = self.load_raw_file(f'test_{dataset_name}.txt')
        rul_df = self.load_rul_file(f'RUL_{dataset_name}.txt')

        return train_df, test_df, rul_df

    def get_dataset_info(self) -> dict:
        """
        Get information about all available datasets.

        Returns:
            Dictionary with dataset characteristics
        """
        return {
            'FD001': {
                'train_engines': 100,
                'test_engines': 100,
                'conditions': 1,
                'fault_modes': 1,
                'description': 'Single operating condition, single fault mode (simplest)'
            },
            'FD002': {
                'train_engines': 260,
                'test_engines': 259,
                'conditions': 6,
                'fault_modes': 1,
                'description': 'Six operating conditions, single fault mode'
            },
            'FD003': {
                'train_engines': 100,
                'test_engines': 100,
                'conditions': 1,
                'fault_modes': 2,
                'description': 'Single operating condition, two fault modes'
            },
            'FD004': {
                'train_engines': 248,
                'test_engines': 249,
                'conditions': 6,
                'fault_modes': 2,
                'description': 'Six operating conditions, two fault modes (most complex)'
            }
        }

    def print_dataset_summary(self, df: pd.DataFrame, dataset_type: str = "Dataset"):
        """
        Print summary statistics for a dataset.

        Args:
            df: DataFrame to summarize
            dataset_type: Type label (e.g., "Training", "Test")
        """
        print(f"\n{'='*70}")
        print(f"{dataset_type.upper()} SUMMARY")
        print(f"{'='*70}")
        print(f"Total samples: {len(df):,}")
        print(f"Engines: {df['engine_id'].nunique()}")
        print(f"Max cycles per engine: {df.groupby('engine_id')['cycle'].max().max()}")
        print(f"Avg cycles per engine: {df.groupby('engine_id')['cycle'].max().mean():.1f}")
        print(f"Sensors: {len([c for c in df.columns if c.startswith('sensor_')])}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"{'='*70}\n")

    def download_all(self) -> bool:
        """
        Main method to download and extract all datasets.

        Returns:
            True if successful, False otherwise
        """
        print("\n" + "="*70)
        print("NASA C-MAPSS TURBOFAN DATASET DOWNLOADER")
        print("="*70)
        print("Course: MLOps - M2 Data Science")
        print("Institution: Aix-Marseille University")
        print("="*70 + "\n")

        # Check for ZIP file
        zip_path = self.raw_dir / "CMAPSSData.zip"

        if not zip_path.exists():
            # Provide manual download instructions
            success = self.download_dataset_manual()
            if not success:
                return False

        # Extract if needed
        if zip_path.exists():
            # Check if already extracted
            if (self.raw_dir / "train_FD001.txt").exists():
                logger.info(" Dataset already extracted")
            else:
                self.extract_zip(zip_path)

        # Verify all files exist
        datasets = ['FD001', 'FD002', 'FD003', 'FD004']
        all_files_exist = True

        for dataset in datasets:
            train_file = self.raw_dir / f"train_{dataset}.txt"
            test_file = self.raw_dir / f"test_{dataset}.txt"
            rul_file = self.raw_dir / f"RUL_{dataset}.txt"

            if not (train_file.exists() and test_file.exists() and rul_file.exists()):
                logger.warning(f" Missing files for {dataset}")
                all_files_exist = False

        if all_files_exist:
            logger.info("\n All dataset files are available!")
            self._print_quick_start()
            return True
        else:
            logger.error("\n Some files are missing. Please complete manual download.")
            return False

    def _print_quick_start(self):
        """Print quick start guide."""
        print("\n" + "="*70)
        print("QUICK START")
        print("="*70)
        print("\nLoad a dataset in Python:")
        print("""
from data.download_data import TurbofanDataDownloader

downloader = TurbofanDataDownloader()
train_df, test_df, rul_df = downloader.load_dataset('FD001')

print(train_df.head())
        """)
        print("="*70 + "\n")


def main():
    """
    Main execution function.

    This function demonstrates the complete download workflow.
    """
    # Initialize downloader
    downloader = TurbofanDataDownloader()

    # Download and extract
    success = downloader.download_all()

    if not success:
        logger.error("Download incomplete. Please follow manual instructions.")
        return

    # Load and display FD001 as example
    try:
        print("\nLoading FD001 dataset as example...")
        train_df, test_df, rul_df = downloader.load_dataset('FD001')

        # Print summaries
        downloader.print_dataset_summary(train_df, "Training Set")
        downloader.print_dataset_summary(test_df, "Test Set")

        # Print RUL info
        print(f"RUL values (first 5 engines):")
        print(rul_df.head())
        print(f"\nRUL statistics:")
        print(rul_df['RUL'].describe())

        # Print dataset information
        print("\n" + "="*70)
        print("ALL DATASETS OVERVIEW")
        print("="*70)
        info = downloader.get_dataset_info()
        for name, details in info.items():
            print(f"\n{name}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
        print("="*70 + "\n")

        logger.info(" Download and verification complete!")

    except FileNotFoundError as e:
        logger.error(f"Error loading dataset: {e}")
        logger.info("Please ensure all files are downloaded correctly.")


if __name__ == "__main__":
    main()
