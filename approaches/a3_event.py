import torch
import numpy as np
import torchaudio.transforms as T

N_MFCC = 13
SAMPLE_RATE = 16000

# load once at import time — small model, negligible overhead
_vad_model, _vad_utils = torch.hub.load(
    'snakers4/silero-vad', 'silero_vad', trust_repo=True, verbose=False
)
_get_speech_timestamps = _vad_utils[0]

# threshold 0.0–1.0: speech probability above which a frame counts as speech
VAD_THRESHOLD = 0.5


def vad_triggered(waveform: torch.Tensor, threshold: float = VAD_THRESHOLD) -> bool:
    timestamps = _get_speech_timestamps(
        waveform.squeeze(), _vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=threshold,
    )
    return len(timestamps) > 0


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
