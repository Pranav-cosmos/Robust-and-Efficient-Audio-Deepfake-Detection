"""MFCC feature extraction with delta and delta-delta computation.

Supports:
    - Utterance-level 240-dim feature vectors (mean + std pooling)
    - Batched GPU/CPU vectorized computation across audio batches
"""
import numpy as np
import torch
import torchaudio
from typing import Optional, Tuple, Union


class MFCCExtractor:
    """Extract MFCC features with deltas and delta-deltas."""

    def __init__(
        self,
        n_mfcc: int = 40,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 40,
        sample_rate: int = 16000,
        compute_deltas: bool = True,
        compute_delta_deltas: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.compute_deltas = compute_deltas
        self.compute_delta_deltas = compute_delta_deltas
        self.device = device or torch.device("cpu")

        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "win_length": win_length,
                "hop_length": hop_length,
                "n_mels": n_mels,
            },
        ).to(self.device)

        self.delta_transform = torchaudio.transforms.ComputeDeltas().to(self.device)

    def extract_batch_utterance_level(self, waveforms: torch.Tensor) -> np.ndarray:
        """Vectorized extraction of utterance-level MFCCs for a batch.

        Args:
            waveforms: Audio tensor (B, 1, T) or (B, T).

        Returns:
            Feature matrix of shape (B, 240) as NumPy array.
        """
        if waveforms.dim() == 2:
            waveforms = waveforms.unsqueeze(1)  # (B, 1, T)
        elif waveforms.dim() == 3 and waveforms.shape[1] > 1:
            waveforms = waveforms.mean(dim=1, keepdim=True)

        waveforms = waveforms.to(self.device)

        with torch.no_grad():
            # (B, 1, n_mfcc, num_frames) -> squeeze channel -> (B, n_mfcc, num_frames)
            mfcc = self.mfcc_transform(waveforms)
            if mfcc.dim() == 4:
                mfcc = mfcc.squeeze(1)

            features = [mfcc]

            if self.compute_deltas:
                delta = self.delta_transform(mfcc)
                features.append(delta)

            if self.compute_delta_deltas:
                if self.compute_deltas:
                    delta_delta = self.delta_transform(delta)
                else:
                    delta = self.delta_transform(mfcc)
                    delta_delta = self.delta_transform(delta)
                features.append(delta_delta)

            # Combined shape: (B, 120, num_frames)
            combined = torch.cat(features, dim=1)

            # Mean and std pooling across frames (dim=2)
            mean_features = combined.mean(dim=2)  # (B, 120)
            std_features = combined.std(dim=2)    # (B, 120)

            # Concatenate -> (B, 240)
            utterance_features = torch.cat([mean_features, std_features], dim=1)

        return utterance_features.cpu().numpy()

    def extract_utterance_level(self, waveform: torch.Tensor) -> np.ndarray:
        """Extract utterance-level features for a single waveform."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        return self.extract_batch_utterance_level(waveform)[0]

    @property
    def frame_feature_dim(self) -> int:
        dim = self.n_mfcc
        if self.compute_deltas:
            dim += self.n_mfcc
        if self.compute_delta_deltas:
            dim += self.n_mfcc
        return dim

    @property
    def utterance_feature_dim(self) -> int:
        return self.frame_feature_dim * 2
