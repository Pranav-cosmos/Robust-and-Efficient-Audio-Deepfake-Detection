"""Simple CNN architecture for audio deepfake detection.

Lightweight 4-block CNN operating on log-mel spectrograms.
Input: (B, 1, 80, T) log mel spectrogram
Output: (B, 2) class logits
"""
import torch
import torch.nn as nn
from typing import Any, Dict


class SimpleCNN(nn.Module):
    """Lightweight 4-block CNN for mel spectrogram classification.

    Architecture:
        4x [Conv2d -> BatchNorm -> ReLU -> MaxPool2d]
        -> AdaptiveAvgPool2d -> Flatten -> FC(128) -> ReLU -> Dropout(0.3) -> FC(2)
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize SimpleCNN.

        Args:
            config: Configuration with model.architecture fields.
        """
        super().__init__()
        arch = config.get("model", {}).get("architecture", {})

        in_channels = arch.get("in_channels", 1)
        channels = arch.get("channels", [16, 32, 64, 128])
        kernel_sizes = arch.get("kernel_sizes", [3, 3, 3, 3])
        pool_sizes = arch.get("pool_sizes", [2, 2, 2, 2])
        fc_hidden = arch.get("fc_hidden", 128)
        num_classes = arch.get("num_classes", 2)
        dropout = arch.get("dropout", 0.3)
        use_bn = arch.get("batch_norm", True)

        # Build convolutional blocks
        layers = []
        prev_channels = in_channels
        for c, k, p in zip(channels, kernel_sizes, pool_sizes):
            layers.append(nn.Conv2d(prev_channels, c, kernel_size=k, padding=k // 2))
            if use_bn:
                layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=p))
            prev_channels = c

        self.features = nn.Sequential(*layers)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

        self.embedding_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, 1, n_mels, num_frames).

        Returns:
            Logits tensor (B, num_classes).
        """
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding from penultimate layer."""
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        return x
