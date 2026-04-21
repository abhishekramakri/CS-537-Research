import torchaudio
import os

DATA_DIR = os.path.dirname(__file__)  # downloads into data/SpeechCommands/

def download():
    print("Downloading Google Speech Commands v2 (~2.3 GB)...")
    train = torchaudio.datasets.SPEECHCOMMANDS(DATA_DIR, subset="training", download=True)
    val   = torchaudio.datasets.SPEECHCOMMANDS(DATA_DIR, subset="validation", download=True)
    test  = torchaudio.datasets.SPEECHCOMMANDS(DATA_DIR, subset="testing", download=True)

    print(f"Train: {len(train)} samples")
    print(f"Val:   {len(val)} samples")
    print(f"Test:  {len(test)} samples")

    waveform, sample_rate, label, *_ = train[0]
    print(f"\nSample: label='{label}', shape={waveform.shape}, sr={sample_rate}")

if __name__ == "__main__":
    download()
