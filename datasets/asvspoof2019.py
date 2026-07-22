"""ASVspoof 2019 LA dataset loader.

Parses official protocol files and provides PyTorch Dataset interface.
Follows official train/dev/eval splits with no speaker or attack leakage.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


class ASVspoof2019LA(Dataset):
    """PyTorch Dataset for ASVspoof 2019 Logical Access (LA).

    Expected directory structure:
        ASVspoof2019_LA/
        ├── ASVspoof2019_LA_cm_protocols/
        │   ├── ASVspoof2019.LA.cm.train.trn.txt
        │   ├── ASVspoof2019.LA.cm.dev.trl.txt
        │   └── ASVspoof2019.LA.cm.eval.trl.txt
        ├── ASVspoof2019_LA_train/
        │   └── flac/
        ├── ASVspoof2019_LA_dev/
        │   └── flac/
        └── ASVspoof2019_LA_eval/
            └── flac/

    Protocol file format (space-separated):
        SPEAKER_ID AUDIO_FILE_ID - ATTACK_TYPE LABEL
        Example: LA_0079 LA_T_1138215 - A07 spoof
    """

    LABEL_MAP = {"bonafide": 0, "spoof": 1}

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        sample_rate: int = 16000,
        max_samples: int = 64000,
        transform=None,
    ):
        """Initialize ASVspoof 2019 LA dataset.

        Args:
            root_dir: Path to ASVspoof2019_LA directory.
            split: One of 'train', 'dev', 'eval'.
            sample_rate: Target sample rate.
            max_samples: Maximum number of audio samples (pad/truncate).
            transform: Optional audio transform.
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self.transform = transform

        # Map split names to protocol/audio directory names
        split_map = {
            "train": ("ASVspoof2019.LA.cm.train.trn.txt", "ASVspoof2019_LA_train"),
            "dev": ("ASVspoof2019.LA.cm.dev.trl.txt", "ASVspoof2019_LA_dev"),
            "eval": ("ASVspoof2019.LA.cm.eval.trl.txt", "ASVspoof2019_LA_eval"),
        }

        if split not in split_map:
            raise ValueError(f"split must be one of {list(split_map.keys())}, got '{split}'")

        protocol_file, audio_dir = split_map[split]
        self.protocol_path = self.root_dir / "ASVspoof2019_LA_cm_protocols" / protocol_file
        self.audio_dir = self.root_dir / audio_dir / "flac"

        # Parse protocol file
        self.samples = self._parse_protocol()

    def _parse_protocol(self) -> List[Dict]:
        """Parse the protocol file and return list of sample dicts."""
        samples = []

        if not self.protocol_path.exists():
            raise FileNotFoundError(
                f"Protocol file not found: {self.protocol_path}. "
                f"Please download ASVspoof 2019 LA dataset."
            )

        with open(self.protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                speaker_id = parts[0]
                audio_id = parts[1]
                # parts[2] is '-' (separator)
                attack_type = parts[3]
                label_str = parts[4]

                audio_path = self.audio_dir / f"{audio_id}.flac"
                label = self.LABEL_MAP.get(label_str, -1)

                if label == -1:
                    continue

                samples.append({
                    "audio_path": str(audio_path),
                    "audio_id": audio_id,
                    "speaker_id": speaker_id,
                    "attack_type": attack_type,
                    "label": label,
                    "label_str": label_str,
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """Get a sample.

        Returns:
            Tuple of (waveform, label, metadata).
            waveform: Tensor of shape (1, max_samples).
            label: 0 for bonafide, 1 for spoof.
            metadata: Dictionary with speaker_id, audio_id, attack_type.
        """
        sample = self.samples[idx]

        # Load audio using soundfile directly to bypass torchaudio torchcodec dependency
        try:
            audio_data, sr = sf.read(sample["audio_path"], dtype="float32", always_2d=False)
            waveform = torch.from_numpy(audio_data).unsqueeze(0)  # (1, T)
        except Exception as e:
            print(f"[WARNING] Could not load {sample['audio_path']}: {e} — returning silence.")
            waveform = torch.zeros(1, self.max_samples)
            sr = self.sample_rate

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Pad or truncate to fixed length
        waveform = self._pad_or_truncate(waveform)

        # Apply transform
        if self.transform is not None:
            waveform = self.transform(waveform)

        metadata = {
            "audio_id": sample["audio_id"],
            "speaker_id": sample["speaker_id"],
            "attack_type": sample["attack_type"],
        }

        return waveform, sample["label"], metadata

    def _pad_or_truncate(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pad or truncate waveform to self.max_samples."""
        num_samples = waveform.shape[-1]
        if num_samples > self.max_samples:
            waveform = waveform[..., :self.max_samples]
        elif num_samples < self.max_samples:
            pad_length = self.max_samples - num_samples
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        return waveform

    def get_label_counts(self) -> Dict[str, int]:
        """Get the number of bonafide and spoof samples."""
        labels = [s["label"] for s in self.samples]
        return {
            "bonafide": labels.count(0),
            "spoof": labels.count(1),
            "total": len(labels),
        }

    def get_speakers(self) -> List[str]:
        """Get list of unique speaker IDs."""
        return list(set(s["speaker_id"] for s in self.samples))

    def get_attack_types(self) -> List[str]:
        """Get list of unique attack types."""
        return list(set(s["attack_type"] for s in self.samples))
