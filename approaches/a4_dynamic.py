import torch
import numpy as np
import struct
import torchaudio.transforms as T

SAMPLE_RATE = 16000

# resolution levels: (n_mfcc, hop_length) — coarser hop = fewer frames = fewer bytes
RESOLUTION_LEVELS = {
    "high":   (40, 160),   # 40 coefficients, 10ms hop
    "medium": (20, 320),   # 20 coefficients, 20ms hop
    "low":    (13, 640),   # 13 coefficients, 40ms hop
}


def extract(waveform: torch.Tensor, resolution: str = "high") -> bytes:
    """
    Extracts MFCCs at the specified resolution level.
    Resolution is chosen by device.py based on available bandwidth.
    The n_mfcc value is packed into the first 4 bytes as a header
    so the server knows how to deserialize the payload.
    """
    n_mfcc, hop_length = RESOLUTION_LEVELS[resolution]

    mfcc_transform = T.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": 400, "hop_length": hop_length, "n_mels": 40}
    )
    mfcc = mfcc_transform(waveform)  # (1, n_mfcc, n_frames)
    mfcc_bytes = mfcc.numpy().astype(np.float32).tobytes()

    # prepend n_mfcc and n_frames as a 2-integer header so server can reshape
    n_frames = mfcc.shape[-1]
    header = struct.pack("!II", n_mfcc, n_frames)
    return header + mfcc_bytes


def deserialize(data: bytes) -> torch.Tensor:
    # read the header to get shape, then reconstruct tensor
    n_mfcc, n_frames = struct.unpack("!II", data[:8])
    arr = np.frombuffer(data[8:], dtype=np.float32).reshape(1, n_mfcc, n_frames)
    return torch.tensor(arr).unsqueeze(0)  # (1, 1, n_mfcc, n_frames)


def payload_size(resolution: str) -> int:
    n_mfcc, hop_length = RESOLUTION_LEVELS[resolution]
    n_frames = 1 + (SAMPLE_RATE - 400) // hop_length
    return 8 + n_mfcc * n_frames * 4  # 8 bytes header + data
