#!/usr/bin/env python3
"""
Turbofan Autoencoder for Anomaly Detection

This module implements a simple but effective autoencoder architecture
for detecting anomalies in turbofan engine sensor data.

Key Concept:
- Train on NORMAL engine operation data
- Learn to reconstruct normal patterns
- High reconstruction error = ANOMALY (potential failure)

Architecture:
- Encoder: Input (21 sensors) → 16 → 8 → 4 (bottleneck)
- Decoder: 4 → 8 → 16 → Output (21 sensors)

Author: Feda Almuhisen
Course: M2 SID - Processus Data
Institution: Aix-Marseille University
Year: 2025-2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TurbofanAutoencoder(nn.Module):
    """
    Autoencoder for turbofan sensor anomaly detection.

    The model learns to compress sensor data into a low-dimensional
    representation (bottleneck) and then reconstruct the original data.

    For NORMAL operation:
        - Reconstruction error is LOW
        - Model learned these patterns during training

    For ANOMALOUS operation:
        - Reconstruction error is HIGH
        - Model hasn't seen these patterns (degradation, faults)

    Architecture:
        Encoder: input_dim → hidden_dim1 → hidden_dim2 → encoding_dim
        Decoder: encoding_dim → hidden_dim2 → hidden_dim1 → input_dim

    Attributes:
        input_dim: Number of input features (sensor count)
        encoding_dim: Size of bottleneck layer
        encoder: Encoder network
        decoder: Decoder network

    Example:
        >>> model = TurbofanAutoencoder(input_dim=21, encoding_dim=4)
        >>> x = torch.randn(32, 21)  # Batch of 32 samples
        >>> reconstructed = model(x)
        >>> error = model.get_reconstruction_error(x)
    """

    def __init__(
        self,
        input_dim: int = 21,
        encoding_dim: int = 4,
        hidden_dim1: int = 16,
        hidden_dim2: int = 8,
        dropout_rate: float = 0.1
    ):
        """
        Initialize autoencoder.

        Args:
            input_dim: Number of input features (sensors)
            encoding_dim: Size of bottleneck layer (compressed representation)
            hidden_dim1: Size of first hidden layer
            hidden_dim2: Size of second hidden layer
            dropout_rate: Dropout probability for regularization
        """
        super(TurbofanAutoencoder, self).__init__()

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout_rate = dropout_rate

        # =================================================================
        # ENCODER: Compress input to low-dimensional representation
        # =================================================================
        self.encoder = nn.Sequential(
            # Layer 1: input_dim → hidden_dim1
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(dropout_rate),

            # Layer 2: hidden_dim1 → hidden_dim2
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim2),
            nn.Dropout(dropout_rate),

            # Layer 3: hidden_dim2 → encoding_dim (bottleneck)
            nn.Linear(hidden_dim2, encoding_dim),
            nn.ReLU()  # No dropout on bottleneck
        )

        # =================================================================
        # DECODER: Reconstruct input from compressed representation
        # =================================================================
        self.decoder = nn.Sequential(
            # Layer 1: encoding_dim → hidden_dim2
            nn.Linear(encoding_dim, hidden_dim2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim2),
            nn.Dropout(dropout_rate),

            # Layer 2: hidden_dim2 → hidden_dim1
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim1),
            nn.Dropout(dropout_rate),

            # Layer 3: hidden_dim1 → input_dim (reconstruction)
            nn.Linear(hidden_dim1, input_dim)
            # No activation on output (regression task)
        )

        # Initialize weights
        self._initialize_weights()

        logger.info(f"Initialized TurbofanAutoencoder: {input_dim}→{hidden_dim1}→{hidden_dim2}→{encoding_dim}")

    def _initialize_weights(self):
        """
        Initialize network weights using Xavier initialization.

        This helps with training stability and convergence.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to compressed representation.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Encoded tensor of shape (batch_size, encoding_dim)
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode compressed representation to reconstruction.

        Args:
            z: Encoded tensor of shape (batch_size, encoding_dim)

        Returns:
            Reconstructed tensor of shape (batch_size, input_dim)
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode then decode.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Reconstructed tensor of shape (batch_size, input_dim)
        """
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

    def get_reconstruction_error(
        self,
        x: torch.Tensor,
        reduction: str = 'none'
    ) -> torch.Tensor:
        """
        Calculate reconstruction error (MSE).

        High error indicates anomaly!

        Args:
            x: Input tensor
            reduction: How to reduce error ('none', 'mean', 'sum')
                - 'none': Return error per sample (for ranking)
                - 'mean': Return average error (for metrics)

        Returns:
            Reconstruction error tensor
        """
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)

            if reduction == 'none':
                # Error per sample (average across features)
                error = torch.mean((x - reconstructed) ** 2, dim=1)
            elif reduction == 'mean':
                # Average error across batch
                error = torch.mean((x - reconstructed) ** 2)
            elif reduction == 'sum':
                # Sum of errors
                error = torch.sum((x - reconstructed) ** 2)
            else:
                raise ValueError(f"Invalid reduction: {reduction}")

        return error

    def get_anomaly_score(
        self,
        x: torch.Tensor,
        threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate anomaly scores and binary predictions.

        Args:
            x: Input tensor
            threshold: Threshold for binary classification (if None, only return scores)

        Returns:
            Tuple of (anomaly_scores, is_anomaly)
            - anomaly_scores: Reconstruction error per sample
            - is_anomaly: Binary predictions (1=anomaly, 0=normal)
        """
        scores = self.get_reconstruction_error(x, reduction='none')

        if threshold is not None:
            is_anomaly = (scores > threshold).float()
        else:
            is_anomaly = None

        return scores, is_anomaly

    def get_feature_importance(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate feature-wise reconstruction errors.

        Shows which sensors contribute most to anomaly.

        Args:
            x: Input tensor

        Returns:
            Feature-wise errors of shape (batch_size, input_dim)
        """
        self.eval()
        with torch.no_grad():
            reconstructed = self.forward(x)
            feature_errors = (x - reconstructed) ** 2

        return feature_errors

    def count_parameters(self) -> Tuple[int, int]:
        """
        Count total and trainable parameters.

        Returns:
            Tuple of (total_params, trainable_params)
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


def print_model_summary(model: TurbofanAutoencoder):
    """
    Print comprehensive model summary.

    Args:
        model: TurbofanAutoencoder instance

    Example:
        >>> model = TurbofanAutoencoder(input_dim=21, encoding_dim=4)
        >>> print_model_summary(model)
    """
    print("\n" + "="*70)
    print("TURBOFAN AUTOENCODER ARCHITECTURE")
    print("="*70)

    print("\n### Model Configuration ###")
    print(f"Input dimension:    {model.input_dim}")
    print(f"Hidden layer 1:     {model.hidden_dim1}")
    print(f"Hidden layer 2:     {model.hidden_dim2}")
    print(f"Encoding dimension: {model.encoding_dim}")
    print(f"Dropout rate:       {model.dropout_rate}")

    print("\n### Encoder ###")
    print(model.encoder)

    print("\n### Decoder ###")
    print(model.decoder)

    print("\n### Parameters ###")
    total_params, trainable_params = model.count_parameters()
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Calculate model size in MB
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024**2

    print(f"Model size:           {size_mb:.2f} MB")

    print("="*70 + "\n")


def test_model():
    """
    Test autoencoder with dummy data.
    """
    print("\n### Testing TurbofanAutoencoder ###\n")

    # Create model
    model = TurbofanAutoencoder(
        input_dim=21,
        encoding_dim=4,
        hidden_dim1=16,
        hidden_dim2=8
    )

    print_model_summary(model)

    # Create dummy batch
    batch_size = 32
    x = torch.randn(batch_size, 21)

    print(f"Input shape: {x.shape}")

    # Test forward pass
    reconstructed = model(x)
    print(f"Reconstructed shape: {reconstructed.shape}")

    # Test encoding
    encoded = model.encode(x)
    print(f"Encoded shape: {encoded.shape}")

    # Test reconstruction error
    error = model.get_reconstruction_error(x, reduction='none')
    print(f"Reconstruction error shape: {error.shape}")
    print(f"Mean reconstruction error: {error.mean():.4f}")

    # Test anomaly detection
    threshold = 0.5
    scores, is_anomaly = model.get_anomaly_score(x, threshold=threshold)
    print(f"\nAnomaly detection (threshold={threshold}):")
    print(f"  Anomalies detected: {is_anomaly.sum().item()}/{batch_size}")

    # Test feature importance
    feature_importance = model.get_feature_importance(x)
    print(f"Feature importance shape: {feature_importance.shape}")

    print("\n All tests passed!")


def main():
    """
    Main execution for testing.
    """
    test_model()


if __name__ == "__main__":
    main()
