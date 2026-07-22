"""ASVspoof 2021 LA and DF dataset loaders.

ASVspoof 2021 is evaluation-only (trained on ASVspoof 2019 training set).
Parses the official keys files for labels.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


class ASVspoof2021LA(Dataset):
    """PyTorch Dataset for ASVspoof 2021 Logical Access (LA) evaluation.

    Expected directory structure:
        ASVspoof2021_LA/
        ├── ASVspoof2021_LA_eval/
        │   └── flac/
        └── keys/
            └── LA/
                └── CM/
                    └── trial_metadata.txt

    Keys file format (space-separated):
        SPEAKER_ID AUDIO_FILE_ID CODEC ATTACK_TYPE LABEL
    """

    LABEL_MAP = {"bonafide": 0, "spoof": 1}

    def __init__(
        self,
        root_dir: str,
        keys_dir: Optional[str] = None,
        sample_rate: int = 16000,
        max_samples: int = 64000,
        transform=None,
        max_files: Optional[int] = None,
    ):
        """Initialize ASVspoof 2021 LA dataset.

        Args:
            root_dir: Path to ASVspoof2021_LA directory.
            keys_dir: Path to keys directory. If None, looks inside root_dir.
            sample_rate: Target sample rate.
            max_samples: Maximum number of audio samples (pad/truncate).
            transform: Optional audio transform.
            max_files: Optional limit on number of files (for debugging).
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self.transform = transform

        # Audio directory
        self.audio_dir = self.root_dir / "ASVspoof2021_LA_eval" / "flac"
        if not self.audio_dir.exists():
            # Try alternative structure
            self.audio_dir = self.root_dir / "flac"

        # Keys file
        if keys_dir is not None:
            keys_path = Path(keys_dir)
        else:
            keys_path = self.root_dir / "keys" / "LA" / "CM"

        self.keys_file = keys_path / "trial_metadata.txt"
        if not self.keys_file.exists():
            # Try alternative name
            self.keys_file = keys_path / "keys" / "trial_metadata.txt"

        self.samples = self._parse_keys(max_files)

    def _parse_keys(self, max_files: Optional[int] = None) -> List[Dict]:
        """Parse keys/metadata file."""
        samples = []

        if not self.keys_file.exists():
            raise FileNotFoundError(
                f"Keys file not found: {self.keys_file}. "
                f"Please download ASVspoof 2021 LA keys."
            )

        with open(self.keys_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                speaker_id = parts[0]
                audio_id = parts[1]
                attack_type = parts[3] if len(parts) > 3 else "-"
                label_str = parts[4] if len(parts) > 4 else parts[-1]

                label = self.LABEL_MAP.get(label_str, -1)
                if label == -1:
                    continue

                audio_path = self.audio_dir / f"{audio_id}.flac"
                if not audio_path.exists():
                    continue

                samples.append({
                    "audio_path": str(audio_path),
                    "audio_id": audio_id,
                    "speaker_id": speaker_id,
                    "attack_type": attack_type,
                    "label": label,
                })

                if max_files and len(samples) >= max_files:
                    break

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        sample = self.samples[idx]

        audio_data, sr = sf.read(sample["audio_path"], dtype="float32", always_2d=False)
        waveform = torch.from_numpy(audio_data).unsqueeze(0)  # (1, T)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = self._pad_or_truncate(waveform)

        if self.transform is not None:
            waveform = self.transform(waveform)

        metadata = {
            "audio_id": sample["audio_id"],
            "speaker_id": sample["speaker_id"],
            "attack_type": sample["attack_type"],
        }

        return waveform, sample["label"], metadata

    def _pad_or_truncate(self, waveform: torch.Tensor) -> torch.Tensor:
        num_samples = waveform.shape[-1]
        if num_samples > self.max_samples:
            waveform = waveform[..., :self.max_samples]
        elif num_samples < self.max_samples:
            pad_length = self.max_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        return waveform


class ASVspoof2021DF(Dataset):
    """PyTorch Dataset for ASVspoof 2021 DeepFake (DF) evaluation.

    Expected directory structure:
        ASVspoof2021_DF/
        ├── ASVspoof2021_DF_eval/
        │   └── flac/
        └── keys/
            └── DF/
                └── CM/
                    └── trial_metadata.txt
    """

    LABEL_MAP = {"bonafide": 0, "spoof": 1}

    def __init__(
        self,
        root_dir: str,
        keys_dir: Optional[str] = None,
        sample_rate: int = 16000,
        max_samples: int = 64000,
        transform=None,
        max_files: Optional[int] = None,
    ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self.transform = transform

        self.audio_dir = self.root_dir / "ASVspoof2021_DF_eval" / "flac"
        if not self.audio_dir.exists():
            self.audio_dir = self.root_dir / "flac"

        if keys_dir is not None:
            keys_path = Path(keys_dir)
        else:
            keys_path = self.root_dir / "keys" / "DF" / "CM"

        self.keys_file = keys_path / "trial_metadata.txt"

        self.samples = self._parse_keys(max_files)

    def _parse_keys(self, max_files: Optional[int] = None) -> List[Dict]:
        samples = []

        if not self.keys_file.exists():
            raise FileNotFoundError(
                f"Keys file not found: {self.keys_file}. "
                f"Please download ASVspoof 2021 DF keys."
            )

        with open(self.keys_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                speaker_id = parts[0]
                audio_id = parts[1]
                attack_type = parts[3] if len(parts) > 3 else "-"
                label_str = parts[4] if len(parts) > 4 else parts[-1]

                label = self.LABEL_MAP.get(label_str, -1)
                if label == -1:
                    continue

                audio_path = self.audio_dir / f"{audio_id}.flac"
                if not audio_path.exists():
                    continue

                samples.append({
                    "audio_path": str(audio_path),
                    "audio_id": audio_id,
                    "speaker_id": speaker_id,
                    "attack_type": attack_type,
                    "label": label,
                })

                if max_files and len(samples) >= max_files:
                    break

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        sample = self.samples[idx]

        audio_data, sr = sf.read(sample["audio_path"], dtype="float32", always_2d=False)
        waveform = torch.from_numpy(audio_data).unsqueeze(0)  # (1, T)


        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = self._pad_or_truncate(waveform)

        if self.transform is not None:
            waveform = self.transform(waveform)

        metadata = {
            "audio_id": sample["audio_id"],
            "speaker_id": sample["speaker_id"],
            "attack_type": sample["attack_type"],
        }

        return waveform, sample["label"], metadata

    def _pad_or_truncate(self, waveform: torch.Tensor) -> torch.Tensor:
        num_samples = waveform.shape[-1]
        if num_samples > self.max_samples:
            waveform = waveform[..., :self.max_samples]
        elif num_samples < self.max_samples:
            pad_length = self.max_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        return waveform
