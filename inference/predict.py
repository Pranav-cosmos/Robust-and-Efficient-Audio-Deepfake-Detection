"""Single-file and batch inference script for audio deepfake detection.

Usage:
    python inference/predict.py --model resnet18 --audio_path test.wav
    python inference/predict.py --model xgboost --audio_path test.wav
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, resolve_model_config
from utils.seed import set_seed, get_device
from preprocessing.audio_preprocessing import load_audio, pad_or_truncate
from features.mfcc_extractor import MFCCExtractor
from features.mel_spectrogram_extractor import MelSpectrogramExtractor

from models.deep.simple_cnn import SimpleCNN
from models.deep.mobilenetv2 import MobileNetV2Audio
from models.deep.resnet18 import ResNet18Audio
from models.classical.random_forest import RandomForestModel
from models.classical.xgboost_model import XGBoostModel
from features.feature_utils import load_scaler

DEEP_MODELS = {
    "simple_cnn": SimpleCNN,
    "mobilenetv2": MobileNetV2Audio,
    "resnet18": ResNet18Audio,
}

CLASSICAL_MODELS = {
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
}

LABEL_NAMES = {0: "bonafide", 1: "spoof"}


def predict_deep(
    model_name: str,
    audio_path: str,
    checkpoint_path: str,
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Run inference with a deep learning model."""
    audio_cfg = config.get("audio", {})
    waveform, sr = load_audio(
        audio_path,
        sample_rate=audio_cfg.get("sample_rate", 16000),
    )
    waveform = pad_or_truncate(waveform, audio_cfg.get("max_samples", 64000))

    model = DEEP_MODELS[model_name](config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    mel_cfg = config.get("mel_spectrogram", {})
    extractor = MelSpectrogramExtractor(
        n_mels=mel_cfg.get("n_mels", 80),
    )
    mel_spec = extractor.extract(waveform)
    input_tensor = mel_spec.unsqueeze(0).to(device)

    start_time = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
    inference_time = (time.perf_counter() - start_time) * 1000

    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0, pred].item()

    return {
        "audio_path": audio_path,
        "prediction": LABEL_NAMES[pred],
        "label_id": pred,
        "confidence": confidence,
        "bonafide_prob": probs[0, 0].item(),
        "spoof_prob": probs[0, 1].item(),
        "inference_time_ms": inference_time,
        "model": model_name,
    }


def predict_classical(
    model_name: str,
    audio_path: str,
    model_path: str,
    scaler_path: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run inference with a classical ML model."""
    audio_cfg = config.get("audio", {})
    waveform, sr = load_audio(
        audio_path,
        sample_rate=audio_cfg.get("sample_rate", 16000),
    )
    waveform = pad_or_truncate(waveform, audio_cfg.get("max_samples", 64000))

    mfcc_cfg = config.get("mfcc", {})
    extractor = MFCCExtractor(n_mfcc=mfcc_cfg.get("n_mfcc", 40))
    features = extractor.extract_utterance_level(waveform)
    features = features.reshape(1, -1)

    scaler = load_scaler(scaler_path)
    features = scaler.transform(features)

    model = CLASSICAL_MODELS[model_name](config)
    model.load(model_path)

    start_time = time.perf_counter()
    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    inference_time = (time.perf_counter() - start_time) * 1000

    return {
        "audio_path": audio_path,
        "prediction": LABEL_NAMES[int(pred)],
        "label_id": int(pred),
        "confidence": float(probs[int(pred)]),
        "bonafide_prob": float(probs[0]),
        "spoof_prob": float(probs[1]),
        "inference_time_ms": inference_time,
        "model": model_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Audio deepfake inference")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(DEEP_MODELS.keys()) + list(CLASSICAL_MODELS.keys()))
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = get_device(args.device)

    checkpoint_dir = config.get("paths", {}).get("checkpoints", "./checkpoints")

    if args.model in DEEP_MODELS:
        checkpoint = args.checkpoint or os.path.join(checkpoint_dir, args.model, "best.pt")
        if not os.path.exists(checkpoint):
            checkpoint = os.path.join(checkpoint_dir, args.model, f"seed_{args.seed}", "best.pt")
        result = predict_deep(args.model, args.audio_path, checkpoint, config, device)
    elif args.model in CLASSICAL_MODELS:
        model_path = args.checkpoint or os.path.join(checkpoint_dir, args.model, f"{args.model}_model.pkl")
        if not os.path.exists(model_path):
            model_path = os.path.join(checkpoint_dir, args.model, f"seed_{args.seed}", f"{args.model}_model.pkl")
        scaler_path = os.path.join(checkpoint_dir, args.model, "scaler.pkl")
        if not os.path.exists(scaler_path):
            scaler_path = os.path.join(checkpoint_dir, args.model, f"seed_{args.seed}", "scaler.pkl")
        result = predict_classical(args.model, args.audio_path, model_path, scaler_path, config)
    else:
        print(f"Unknown model: {args.model}")
        sys.exit(1)

    print("\n=== Prediction Result ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
