import torch
import numpy as np
import torchaudio.transforms as T

N_MFCC = 13
SAMPLE_RATE = 16000

# ratio of peak frame energy to noise floor — relative to clip's own loudness,
# so it works regardless of recording volume. 15 = peak must be 15x the noise floor.
VAD_THRESHOLD = 15.0


def vad_triggered(waveform: torch.Tensor, threshold: float = VAD_THRESHOLD) -> bool:
    frame_size = 400  # 25ms at 16kHz
    frames = waveform.squeeze().unfold(0, frame_size, frame_size // 2)
    energy = (frames ** 2).mean(dim=1)
    noise_floor = max(torch.quantile(energy, 0.2).item(), 1e-8)
    return bool((energy.max().item() / noise_floor) > threshold)


def extract(waveform: torch.Tensor, threshold: float = VAD_THRESHOLD):
    """
    Returns MFCC bytes if voice is detected, None otherwise.
    Caller (device.py) skips transmission when None is returned.
    """
    if not vad_triggered(waveform, threshold):
        return None

    mfcc_transform = T.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
    )
    mfcc = mfcc_transform(waveform)  # (1, 13, 101)
    return mfcc.numpy().astype(np.float32).tobytes()


def deserialize(data: bytes) -> torch.Tensor:
    arr = np.frombuffer(data, dtype=np.float32).reshape(1, N_MFCC, -1)
    return torch.tensor(arr).unsqueeze(0)  # (1, 1, 13, 101)
