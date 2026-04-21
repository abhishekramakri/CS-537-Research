import torch
import numpy as np
import torchaudio.transforms as T

SAMPLE_RATE = 16000


def extract(waveform: torch.Tensor, encoder: torch.nn.Module) -> bytes:
    """
    Runs the encoder on the device side and returns the embedding as bytes.
    encoder must already be loaded and in eval mode before calling this.
    """
    mfcc_transform = T.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=40,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
    )
    mfcc = mfcc_transform(waveform).unsqueeze(0)  # (1, 1, 40, 101)

    with torch.no_grad():
        embedding = encoder(mfcc)  # (1, embedding_dim)

    return embedding.squeeze().numpy().astype(np.float32).tobytes()


def deserialize(data: bytes, embedding_dim: int) -> torch.Tensor:
    arr = np.frombuffer(data, dtype=np.float32).reshape(1, embedding_dim)
    return torch.tensor(arr)  # (1, embedding_dim) — goes straight into classifier


def payload_size(embedding_dim: int) -> int:
    return embedding_dim * 4  # embedding_dim floats * 4 bytes each
