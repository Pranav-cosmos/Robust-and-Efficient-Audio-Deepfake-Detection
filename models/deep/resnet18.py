"""ResNet18 adapted for single-channel mel spectrogram input with transfer learning."""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from typing import Any, Dict


class ResNet18Audio(nn.Module):
    """ResNet18 adapted for single-channel audio spectrogram classification.

    Modifications:
        - Accepts 1-channel log-mel spectrogram input.
        - Uses ImageNet pretrained weights.
        - Freezes initial conv layer and layer1 for fast, memory-efficient transfer learning.
        - Outputs 2-class logits.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize ResNet18 for audio.

        Args:
            config: Configuration dictionary with model.architecture fields.
        """
        super().__init__()
        arch = config.get("model", {}).get("architecture", {})

        in_channels = arch.get("in_channels", 1)
        num_classes = arch.get("num_classes", 2)
        use_pretrained = arch.get("pretrained", True)

        weights = ResNet18_Weights.DEFAULT if use_pretrained else None
        self.backbone = resnet18(weights=weights)

        # Modify first conv for 1-channel input while preserving pretrained weight sum
        original_conv1 = self.backbone.conv1
        new_conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        if use_pretrained:
            with torch.no_grad():
                new_conv1.weight.copy_(original_conv1.weight.sum(dim=1, keepdim=True))

        self.backbone.conv1 = new_conv1

        # Freeze initial conv block and layer1 for efficient fine-tuning
        if use_pretrained:
            for param in self.backbone.conv1.parameters():
                param.requires_grad = False
            for param in self.backbone.bn1.parameters():
                param.requires_grad = False
            for param in self.backbone.layer1.parameters():
                param.requires_grad = False

        # Modify final FC layer
        self.backbone.fc = nn.Linear(512, num_classes)

        self.embedding_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.backbone(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 512-dim embedding from avgpool layer."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x
