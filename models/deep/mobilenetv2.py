"""MobileNetV2 adapted for single-channel mel spectrogram input with transfer learning."""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from typing import Any, Dict


class MobileNetV2Audio(nn.Module):
    """MobileNetV2 adapted for single-channel audio spectrogram classification.

    Modifications:
        - Accepts 1-channel log-mel spectrogram input.
        - Option for ImageNet pretrained weights.
        - Freezes lower feature layers (features[0:6]) for fast convergence and efficiency.
        - Modifies classifier head for binary classification.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize MobileNetV2 for audio.

        Args:
            config: Configuration dictionary with model.architecture fields.
        """
        super().__init__()
        arch = config.get("model", {}).get("architecture", {})

        in_channels = arch.get("in_channels", 1)
        num_classes = arch.get("num_classes", 2)
        width_mult = arch.get("width_mult", 1.0)
        dropout = arch.get("dropout", 0.2)
        use_pretrained = arch.get("pretrained", True)

        weights = MobileNet_V2_Weights.DEFAULT if use_pretrained else None
        self.backbone = mobilenet_v2(weights=weights, width_mult=width_mult)

        # Modify first conv layer for 1-channel input while preserving pretrained weight sum
        original_first_conv = self.backbone.features[0][0]
        new_first_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=False,
        )
        if use_pretrained:
            with torch.no_grad():
                new_first_conv.weight.copy_(original_first_conv.weight.sum(dim=1, keepdim=True))

        self.backbone.features[0][0] = new_first_conv

        # Freeze early feature blocks (features[0:6]) for efficient fine-tuning
        if use_pretrained:
            for i in range(6):
                for param in self.backbone.features[i].parameters():
                    param.requires_grad = False

        # Modify classifier head
        last_channel = self.backbone.last_channel
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channel, num_classes),
        )

        self.embedding_dim = last_channel  # 1280 for width_mult=1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding before classification head."""
        x = self.backbone.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x
