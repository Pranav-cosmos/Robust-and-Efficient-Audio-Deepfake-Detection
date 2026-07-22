"""WaveFake dataset loader.

WaveFake contains synthetic audio from multiple TTS/vocoder architectures.
LJSpeech provides the bonafide reference audio.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


class WaveFakeDataset(Dataset):
    """PyTorch Dataset for WaveFake.

    Expected directory structure:
        WaveFake/
        ├── ljspeech_full_band_melgan/
        ├── ljspeech_hifiGAN/
        ├── ljspeech_melgan/
        ├── ljspeech_melgan_large/
        ├── ljspeech_multi_band_melgan/
        ├── ljspeech_parallel_wavegan/
        └── ljspeech_waveglow/

        LJSpeech-1.1/
        └── wavs/

    All spoof audio is in the WaveFake subfolders.
    Bonafide audio comes from LJSpeech-1.1/wavs/.
    """

    def __init__(
        self,
        wavefake_dir: str,
        ljspeech_dir: str,
        sample_rate: int = 16000,
        max_samples: int = 64000,
        transform=None,
        max_files_per_source: Optional[int] = None,
    ):
        """Initialize WaveFake dataset.

        Args:
            wavefake_dir: Path to WaveFake directory.
            ljspeech_dir: Path to LJSpeech-1.1 directory.
            sample_rate: Target sample rate.
            max_samples: Maximum audio samples (pad/truncate).
            transform: Optional audio transform.
            max_files_per_source: Limit files per source (for debugging).
        """
        super().__init__()
        self.wavefake_dir = Path(wavefake_dir)
        self.ljspeech_dir = Path(ljspeech_dir)
        self.sample_rate = sample_rate
        self.max_samples = max_samples
        self.transform = transform

        self.samples = self._scan_files(max_files_per_source)

    def _scan_files(self, max_per_source: Optional[int] = None) -> List[Dict]:
        """Scan directory structure to build sample list."""
        samples = []

        # Bonafide from LJSpeech
        ljspeech_wavs = self.ljspeech_dir / "wavs"
        if ljspeech_wavs.exists():
            wav_files = sorted(ljspeech_wavs.glob("*.wav"))
            if max_per_source:
                wav_files = wav_files[:max_per_source]

            for wav_path in wav_files:
                samples.append({
                    "audio_path": str(wav_path),
                    "audio_id": wav_path.stem,
                    "source": "ljspeech",
                    "attack_type": "-",
                    "label": 0,  # bonafide
                })

        # Spoof from WaveFake subdirectories
        if self.wavefake_dir.exists():
            for subdir in sorted(self.wavefake_dir.iterdir()):
                if not subdir.is_dir():
                    continue

                source_name = subdir.name
                wav_files = sorted(list(subdir.glob("*.wav")))
                if max_per_source:
                    wav_files = wav_files[:max_per_source]

                for wav_path in wav_files:
                    samples.append({
                        "audio_path": str(wav_path),
                        "audio_id": wav_path.stem,
                        "source": source_name,
                        "attack_type": source_name,
                        "label": 1,  # spoof
                    })

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
            "source": sample["source"],
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

    def get_sources(self) -> List[str]:
        """Get list of unique audio sources."""
        return list(set(s["source"] for s in self.samples))

    def get_label_counts(self) -> Dict[str, int]:
        labels = [s["label"] for s in self.samples]
        return {
            "bonafide": labels.count(0),
            "spoof": labels.count(1),
            "total": len(labels),
        }
