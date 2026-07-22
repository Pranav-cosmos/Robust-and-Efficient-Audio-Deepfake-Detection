"""Deep learning models package."""
from models.deep.simple_cnn import SimpleCNN
from models.deep.mobilenetv2 import MobileNetV2Audio
from models.deep.resnet18 import ResNet18Audio

__all__ = ["SimpleCNN", "MobileNetV2Audio", "ResNet18Audio"]
