import torch
import numpy as np
import torchaudio.transforms as T

# fixed 13-coefficient MFCCs — the classic configuration from the literature
N_MFCC = 13
SAMPLE_RATE = 16000


def extract(waveform: torch.Tensor) -> bytes:
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


def payload_size() -> int:
    # 13 coefficients * 101 frames * 4 bytes per float32
    n_frames = 1 + (SAMPLE_RATE - 400) // 160
    return N_MFCC * n_frames * 4
