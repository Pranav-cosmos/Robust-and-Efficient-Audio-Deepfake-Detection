"""Log Mel Spectrogram feature extraction for CNN-based models."""
import torch
import torchaudio
from typing import Optional


class MelSpectrogramExtractor:
    """Extract log-scaled mel spectrograms for CNN input.

    Configuration:
        n_mels: 80
        n_fft: 512
        win_length: 400 (25 ms at 16 kHz)
        hop_length: 160 (10 ms at 16 kHz)
        f_min: 20 Hz
        f_max: 8000 Hz

    Output shape: (1, n_mels, num_frames) suitable for CNN input.
    """

    def __init__(
        self,
        n_mels: int = 80,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        sample_rate: int = 16000,
        log_offset: float = 1e-6,
    ):
        """Initialize mel spectrogram extractor.

        Args:
            n_mels: Number of mel filter banks.
            n_fft: FFT size.
            win_length: Window length in samples.
            hop_length: Hop length in samples.
            f_min: Minimum frequency.
            f_max: Maximum frequency.
            sample_rate: Audio sample rate.
            log_offset: Small value added before log to avoid log(0).
        """
        self.log_offset = log_offset
        self.sample_rate = sample_rate

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract log mel spectrogram.

        Args:
            waveform: Audio tensor (1, num_samples) or (num_samples,).

        Returns:
            Log mel spectrogram of shape (1, n_mels, num_frames).
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Mel spectrogram: (1, n_mels, num_frames)
        mel_spec = self.mel_transform(waveform)

        # Log scale
        log_mel_spec = torch.log(mel_spec + self.log_offset)

        return log_mel_spec

    def extract_batch(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Extract log mel spectrograms for a batch.

        Args:
            waveforms: Batch of waveforms (batch_size, 1, num_samples).

        Returns:
            Log mel spectrograms (batch_size, 1, n_mels, num_frames).
        """
        if waveforms.dim() == 2:
            waveforms = waveforms.unsqueeze(1)

        batch_specs = []
        for i in range(waveforms.shape[0]):
            spec = self.extract(waveforms[i])
            batch_specs.append(spec)

        return torch.stack(batch_specs, dim=0)


class OnTheFlyMelSpectrogram(torch.nn.Module):
    """Mel spectrogram extraction as a PyTorch module for use in data pipelines.

    Can be placed on GPU for accelerated feature extraction.
    """

    def __init__(
        self,
        n_mels: int = 80,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        f_min: float = 20.0,
        f_max: float = 8000.0,
        sample_rate: int = 16000,
        log_offset: float = 1e-6,
    ):
        super().__init__()
        self.log_offset = log_offset

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Forward pass: waveform (B, 1, T) -> log mel (B, 1, n_mels, num_frames)."""
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        # Process each in batch
        specs = []
        for i in range(waveform.shape[0]):
            mel = self.mel_transform(waveform[i])
            log_mel = torch.log(mel + self.log_offset)
            specs.append(log_mel)

        return torch.stack(specs, dim=0)
