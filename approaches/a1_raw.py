import torch
import numpy as np
import torchaudio.transforms as T

# raw 16-bit PCM at 16kHz — this is the baseline, maximum bandwidth
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # 16-bit


def extract(waveform: torch.Tensor) -> bytes:
    # convert float32 waveform [-1, 1] to 16-bit integers, then raw bytes
    pcm = (waveform.squeeze().numpy() * 32767).astype(np.int16)
    return pcm.tobytes()


def deserialize(data: bytes) -> torch.Tensor:
    # reconstruct float32 waveform from PCM bytes, then extract MFCCs
    # server does the MFCC extraction in this approach
    pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
    waveform = torch.tensor(pcm).unsqueeze(0)  # (1, 16000)

    mfcc_transform = T.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=40,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
    )
    mfcc = mfcc_transform(waveform)  # (1, 40, 101)
    return mfcc.unsqueeze(0)         # (1, 1, 40, 101) — batch dim for model


def payload_size(waveform: torch.Tensor) -> int:
    return SAMPLE_RATE * BYTES_PER_SAMPLE  # 32000 bytes per utterance
