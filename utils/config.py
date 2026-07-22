"""Configuration loading with YAML parsing and CLI override support."""
import argparse
import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Dictionary of configuration values.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config if config is not None else {}


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override config into base config.

    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.

    Returns:
        Merged configuration dictionary.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(
    base_config_path: str = "configs/base_config.yaml",
    model_config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load and merge base config, model config, and CLI overrides.

    Args:
        base_config_path: Path to the base configuration file.
        model_config_path: Optional path to model-specific configuration.
        overrides: Optional dictionary of override values.

    Returns:
        Final merged configuration dictionary.
    """
    config = load_yaml_config(base_config_path)

    if model_config_path is not None:
        model_config = load_yaml_config(model_config_path)
        config = merge_configs(config, model_config)

    if overrides is not None:
        config = merge_configs(config, overrides)

    return config


def resolve_paths(config: Dict[str, Any], project_root: Optional[str] = None) -> Dict[str, Any]:
    """Resolve relative paths in config to absolute paths.

    Args:
        config: Configuration dictionary.
        project_root: Project root directory. If None, uses current working directory.

    Returns:
        Config with resolved absolute paths.
    """
    if project_root is None:
        project_root = os.getcwd()

    if "paths" in config:
        for key, value in config["paths"].items():
            if isinstance(value, str) and not os.path.isabs(value):
                config["paths"][key] = os.path.join(project_root, value)

    return config


def get_common_parser() -> argparse.ArgumentParser:
    """Create common argument parser with standard arguments.

    Returns:
        ArgumentParser with common arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                        help="Path to base config file")
    parser.add_argument("--model_config", type=str, default=None,
                        help="Path to model-specific config file")
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (auto-resolves to config file)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cuda, cpu")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode (fewer epochs, less data)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    return parser


def resolve_model_config(model_name: str, config_dir: str = "configs/model_configs") -> str:
    """Resolve model name to config file path.

    Anchors the config_dir relative to this file's project root so that
    scripts can be run from any working directory.

    Args:
        model_name: Short model name (e.g., 'resnet18').
        config_dir: Directory containing model config files, relative to
            the project root (parent of utils/).

    Returns:
        Path to the model config YAML file.

    Raises:
        FileNotFoundError: If no config file exists for the model.
    """
    # Anchor to project root (parent of utils/) regardless of CWD
    project_root = Path(__file__).parent.parent
    abs_config_dir = project_root / config_dir
    config_path = abs_config_dir / f"{model_name}.yaml"
    if not config_path.exists():
        available = list(abs_config_dir.glob("*.yaml"))
        raise FileNotFoundError(
            f"No config file found for model '{model_name}' at {config_path}. "
            f"Available configs: {[p.stem for p in available]}"
        )
    return str(config_path)
