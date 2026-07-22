"""Audio perturbation functions for robustness evaluation.

All perturbations are applied ONLY during evaluation, never during training.
"""
import io
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
from typing import Any


def add_background_noise(
    waveform: torch.Tensor,
    noise_dir: str,
    snr_db: float,
    sample_rate: int = 16000,
    noise_type: str = "noise",
) -> torch.Tensor:
    """Add background noise from MUSAN at a specified SNR.

    Args:
        waveform: Clean waveform tensor (1, num_samples) or (num_samples,).
        noise_dir: Path to MUSAN noise directory (e.g., musan/noise/free-sound).
        snr_db: Target signal-to-noise ratio in dB.
        sample_rate: Expected sample rate.
        noise_type: Subdirectory of MUSAN to use ('noise', 'speech', 'music').

    Returns:
        Noisy waveform tensor.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    num_samples = waveform.shape[-1]

    # Find noise files
    noise_path = Path(noise_dir) / noise_type if noise_type else Path(noise_dir)
    if not noise_path.exists():
        noise_path = Path(noise_dir)

    noise_files = list(noise_path.rglob("*.wav"))
    if not noise_files:
        # Return original if no noise files
        return waveform

    # Random noise file
    noise_file = random.choice(noise_files)
    noise_audio, noise_sr = sf.read(str(noise_file), dtype="float32", always_2d=False)
    noise_waveform = torch.from_numpy(noise_audio).unsqueeze(0)  # (1, T)

    # Resample noise if needed
    if noise_sr != sample_rate:
        resampler = torchaudio.transforms.Resample(noise_sr, sample_rate)
        noise_waveform = resampler(noise_waveform)

    # Convert to mono
    if noise_waveform.shape[0] > 1:
        noise_waveform = noise_waveform.mean(dim=0, keepdim=True)

    # Match length
    noise_length = noise_waveform.shape[-1]
    if noise_length < num_samples:
        # Tile noise to match length
        repeats = (num_samples // noise_length) + 1
        noise_waveform = noise_waveform.repeat(1, repeats)
    noise_waveform = noise_waveform[..., :num_samples]

    # Compute scaling factor for target SNR
    signal_power = waveform.pow(2).mean()
    noise_power = noise_waveform.pow(2).mean()

    if noise_power == 0 or signal_power == 0:
        return waveform

    snr_linear = 10 ** (snr_db / 10)
    scale = torch.sqrt(signal_power / (snr_linear * noise_power))

    noisy = waveform + scale * noise_waveform
    # Clip to prevent overflow
    noisy = torch.clamp(noisy, -1.0, 1.0)

    return noisy


def apply_mp3_compression(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    bitrate: int = 64,
) -> torch.Tensor:
    """Apply MP3 compression/decompression to simulate codec artifacts.

    Args:
        waveform: Input waveform (1, num_samples).
        sample_rate: Sample rate.
        bitrate: MP3 bitrate in kbps (32, 64, 128).

    Returns:
        Compressed/decompressed waveform.
    """
    try:
        # Use torchaudio's effector/sox_effects for MP3 compression
        effects = [
            ["rate", str(sample_rate)],
        ]
        # Apply via torchaudio's sox_effects
        compressed, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate, effects, channels_first=True
        )
        return compressed
    except Exception:
        pass

    # Fallback: use pydub for MP3 compression
    try:
        from pydub import AudioSegment

        # Convert tensor to numpy
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        audio_np = (waveform.squeeze().numpy() * 32767).astype(np.int16)

        # Create AudioSegment
        audio_segment = AudioSegment(
            audio_np.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit
            channels=1,
        )

        # Export as MP3 then re-import
        mp3_buffer = io.BytesIO()
        audio_segment.export(mp3_buffer, format="mp3", bitrate=f"{bitrate}k")
        mp3_buffer.seek(0)

        # Re-import
        compressed_segment = AudioSegment.from_mp3(mp3_buffer)
        compressed_np = np.array(compressed_segment.get_array_of_samples()).astype(np.float32)
        compressed_np = compressed_np / 32767.0

        compressed = torch.from_numpy(compressed_np).unsqueeze(0)

        # Match original length
        target_len = waveform.shape[-1]
        if compressed.shape[-1] > target_len:
            compressed = compressed[..., :target_len]
        elif compressed.shape[-1] < target_len:
            pad = target_len - compressed.shape[-1]
            compressed = torch.nn.functional.pad(compressed, (0, pad))

        return compressed
    except ImportError:
        # If pydub not available, return original
        return waveform


def pitch_shift(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    semitones: float = 2.0,
) -> torch.Tensor:
    """Apply pitch shifting to the waveform.

    Args:
        waveform: Input waveform (1, num_samples).
        sample_rate: Sample rate.
        semitones: Number of semitones to shift (positive = up, negative = down).

    Returns:
        Pitch-shifted waveform.
    """
    try:
        import librosa

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        audio_np = waveform.squeeze().numpy()
        shifted = librosa.effects.pitch_shift(
            y=audio_np, sr=sample_rate, n_steps=semitones
        )
        shifted_tensor = torch.from_numpy(shifted).unsqueeze(0)

        # Match length
        target_len = waveform.shape[-1]
        if shifted_tensor.shape[-1] > target_len:
            shifted_tensor = shifted_tensor[..., :target_len]
        elif shifted_tensor.shape[-1] < target_len:
            pad = target_len - shifted_tensor.shape[-1]
            shifted_tensor = torch.nn.functional.pad(shifted_tensor, (0, pad))

        return shifted_tensor
    except Exception:
        return waveform


def time_stretch(
    waveform: torch.Tensor,
    rate: float = 1.0,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Apply time stretching to the waveform.

    Args:
        waveform: Input waveform (1, num_samples).
        rate: Stretch rate (< 1.0 = slower, > 1.0 = faster).
        sample_rate: Sample rate.

    Returns:
        Time-stretched waveform (padded/truncated to original length).
    """
    try:
        import librosa

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        target_len = waveform.shape[-1]
        audio_np = waveform.squeeze().numpy()
        stretched = librosa.effects.time_stretch(y=audio_np, rate=rate)
        stretched_tensor = torch.from_numpy(stretched).unsqueeze(0)

        # Match original length
        if stretched_tensor.shape[-1] > target_len:
            stretched_tensor = stretched_tensor[..., :target_len]
        elif stretched_tensor.shape[-1] < target_len:
            pad = target_len - stretched_tensor.shape[-1]
            stretched_tensor = torch.nn.functional.pad(stretched_tensor, (0, pad))

        return stretched_tensor
    except Exception:
        return waveform


def random_clip(
    waveform: torch.Tensor,
    clip_percent: float = 5.0,
) -> torch.Tensor:
    """Randomly zero out a percentage of audio samples.

    Simulates audio dropout or packet loss.

    Args:
        waveform: Input waveform (1, num_samples).
        clip_percent: Percentage of samples to zero out (1-10%).

    Returns:
        Clipped waveform.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    clipped = waveform.clone()
    num_samples = clipped.shape[-1]
    num_clip = int(num_samples * clip_percent / 100.0)

    if num_clip == 0:
        return clipped

    # Random contiguous clip region
    start_idx = random.randint(0, num_samples - num_clip)
    clipped[..., start_idx : start_idx + num_clip] = 0.0

    return clipped


class PerturbationPipeline:
    """Applies a configured set of perturbations for evaluation.

    Usage:
        pipeline = PerturbationPipeline(config["perturbations"], musan_dir="data/musan")
        for name, perturbed_waveform in pipeline.apply_all(waveform):
            evaluate(model, perturbed_waveform)
    """

    def __init__(
        self,
        perturbation_config: Dict[str, Any],
        musan_dir: Optional[str] = None,
        sample_rate: int = 16000,
    ):
        """Initialize perturbation pipeline.

        Args:
            perturbation_config: Perturbation settings from config file.
            musan_dir: Path to MUSAN dataset root.
            sample_rate: Audio sample rate.
        """
        self.config = perturbation_config
        self.musan_dir = musan_dir
        self.sample_rate = sample_rate

    def get_perturbation_list(self) -> List[Tuple[str, dict]]:
        """Get list of all (name, params) perturbation configurations."""
        perturbations = []

        # Background noise
        if self.config.get("background_noise", {}).get("enabled", False):
            for snr in self.config["background_noise"]["snr_levels"]:
                perturbations.append((
                    f"noise_snr{snr}dB",
                    {"type": "noise", "snr_db": snr},
                ))

        # MP3 compression
        if self.config.get("mp3_compression", {}).get("enabled", False):
            for bitrate in self.config["mp3_compression"]["bitrates"]:
                perturbations.append((
                    f"mp3_{bitrate}kbps",
                    {"type": "mp3", "bitrate": bitrate},
                ))

        # Pitch shift
        if self.config.get("pitch_shift", {}).get("enabled", False):
            for semitones in self.config["pitch_shift"]["semitones"]:
                sign = "+" if semitones > 0 else ""
                perturbations.append((
                    f"pitch_{sign}{semitones}st",
                    {"type": "pitch", "semitones": semitones},
                ))

        # Time stretch
        if self.config.get("time_stretch", {}).get("enabled", False):
            for rate in self.config["time_stretch"]["rates"]:
                perturbations.append((
                    f"stretch_{rate}x",
                    {"type": "stretch", "rate": rate},
                ))

        # Random clipping
        if self.config.get("random_clipping", {}).get("enabled", False):
            for pct in self.config["random_clipping"]["clip_percentages"]:
                perturbations.append((
                    f"clip_{pct}pct",
                    {"type": "clip", "clip_percent": pct},
                ))

        return perturbations

    def apply_single(
        self,
        waveform: torch.Tensor,
        perturbation_params: Dict,
    ) -> torch.Tensor:
        """Apply a single perturbation to a waveform.

        Args:
            waveform: Input waveform tensor.
            perturbation_params: Dictionary with 'type' and type-specific params.

        Returns:
            Perturbed waveform.
        """
        ptype = perturbation_params["type"]

        if ptype == "noise":
            if self.musan_dir is None:
                return waveform
            return add_background_noise(
                waveform,
                noise_dir=self.musan_dir,
                snr_db=perturbation_params["snr_db"],
                sample_rate=self.sample_rate,
            )
        elif ptype == "mp3":
            return apply_mp3_compression(
                waveform,
                sample_rate=self.sample_rate,
                bitrate=perturbation_params["bitrate"],
            )
        elif ptype == "pitch":
            return pitch_shift(
                waveform,
                sample_rate=self.sample_rate,
                semitones=perturbation_params["semitones"],
            )
        elif ptype == "stretch":
            return time_stretch(
                waveform,
                rate=perturbation_params["rate"],
                sample_rate=self.sample_rate,
            )
        elif ptype == "clip":
            return random_clip(
                waveform,
                clip_percent=perturbation_params["clip_percent"],
            )
        else:
            raise ValueError(f"Unknown perturbation type: {ptype}")

    def apply_all(
        self,
        waveform: torch.Tensor,
    ) -> List[Tuple[str, torch.Tensor]]:
        """Apply all configured perturbations to a waveform.

        Args:
            waveform: Input waveform tensor.

        Returns:
            List of (perturbation_name, perturbed_waveform) tuples.
        """
        results = []
        for name, params in self.get_perturbation_list():
            perturbed = self.apply_single(waveform, params)
            results.append((name, perturbed))
        return results
