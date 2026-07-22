"""Audio preprocessing utilities."""
import numpy as np
import soundfile as sf
import torch
import torchaudio
from typing import Optional, Tuple


def load_audio(
    file_path: str,
    sample_rate: int = 16000,
    mono: bool = True,
    normalize: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Load and preprocess an audio file.

    Args:
        file_path: Path to the audio file.
        sample_rate: Target sample rate.
        mono: Convert to mono if True.
        normalize: Peak-normalize to [-1, 1] if True.

    Returns:
        Tuple of (waveform tensor, sample_rate).
    """
    audio_data, sr = sf.read(file_path, dtype="float32", always_2d=False)
    waveform = torch.from_numpy(audio_data).unsqueeze(0)  # (1, T)

    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)

    # Convert to mono
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Peak normalize
    if normalize:
        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val

    return waveform, sample_rate


def pad_or_truncate(
    waveform: torch.Tensor,
    target_length: int,
    pad_mode: str = "constant",
    pad_value: float = 0.0,
) -> torch.Tensor:
    """Pad or truncate waveform to a fixed length.

    Args:
        waveform: Input waveform tensor (..., num_samples).
        target_length: Target number of samples.
        pad_mode: Padding mode ('constant', 'reflect', 'replicate').
        pad_value: Padding value for 'constant' mode.

    Returns:
        Waveform of shape (..., target_length).
    """
    current_length = waveform.shape[-1]

    if current_length > target_length:
        waveform = waveform[..., :target_length]
    elif current_length < target_length:
        pad_amount = target_length - current_length
        waveform = torch.nn.functional.pad(
            waveform, (0, pad_amount), mode=pad_mode, value=pad_value
        )

    return waveform


def trim_silence(
    waveform: torch.Tensor,
    threshold_db: float = -40.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """Trim leading and trailing silence from waveform.

    Uses energy-based voice activity detection.

    Args:
        waveform: Input waveform (1, num_samples) or (num_samples,).
        threshold_db: Silence threshold in dB.
        frame_length: Analysis frame length.
        hop_length: Hop length for analysis frames.

    Returns:
        Trimmed waveform.
    """
    # Use torchaudio's VAD if available
    try:
        trimmed = torchaudio.functional.vad(
            waveform, sample_rate=16000, trigger_level=abs(threshold_db)
        )
        return trimmed
    except Exception:
        # Fallback: simple energy-based trimming
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        abs_wav = waveform.abs().squeeze()
        threshold = abs_wav.max() * (10 ** (threshold_db / 20))

        above_threshold = abs_wav > threshold
        if not above_threshold.any():
            return waveform

        nonzero = torch.nonzero(above_threshold).squeeze()
        if nonzero.dim() == 0:
            return waveform

        start = int(nonzero[0])
        end = int(nonzero[-1]) + 1

        return waveform[..., start:end]
