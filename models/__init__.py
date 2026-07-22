"""Model definitions and registry for audio deepfake detection."""
from typing import Dict, Any
import torch.nn as nn

from models.classical.random_forest import RandomForestModel
from models.classical.xgboost_model import XGBoostModel
from models.deep.simple_cnn import SimpleCNN
from models.deep.mobilenetv2 import MobileNetV2Audio
from models.deep.resnet18 import ResNet18Audio

MODEL_REGISTRY = {
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "simple_cnn": SimpleCNN,
    "mobilenetv2": MobileNetV2Audio,
    "resnet18": ResNet18Audio,
}


def get_model(name: str, config: Dict[str, Any], **kwargs) -> Any:
    """Instantiate a model by name.

    Args:
        name: Model name.
        config: Model configuration dictionary.

    Returns:
        Model instance.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{name}' not registered. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](config, **kwargs)
