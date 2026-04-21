import os
import argparse
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader

from models.cnn import KWSModel
from models.encoder import Encoder, EmbeddingClassifier

DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3

# all 35 classes in Speech Commands v2, sorted so the index mapping is stable
LABELS = sorted([
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
    "follow", "forward", "four", "go", "happy", "house", "learn", "left",
    "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila",
    "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero"
])
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}


class SpeechCommandsDataset(torch.utils.data.Dataset):
    """
    Wraps the raw torchaudio dataset to return (mfcc_tensor, label_index).
    Also pads/trims each clip to exactly 16000 samples (1 second) since
    a handful of clips in the dataset are slightly shorter.
    """
    def __init__(self, subset, mfcc_transform):
        self.data = torchaudio.datasets.SPEECHCOMMANDS(DATA_DIR, subset=subset, download=False)
        self.transform = mfcc_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveform, _, label, *_ = self.data[idx]

        if waveform.shape[-1] < 16000:
            waveform = nn.functional.pad(waveform, (0, 16000 - waveform.shape[-1]))
        else:
            waveform = waveform[:, :16000]

        mfcc = self.transform(waveform)  # shape: (1, 40, 101)
        return mfcc, LABEL_TO_IDX[label]


def get_mfcc_transform():
    # 40 coefficients, 25ms window (400 samples), 10ms hop (160 samples)
    return T.MFCC(
        sample_rate=16000,
        n_mfcc=40,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
    )


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for mfcc, labels in loader:
            mfcc, labels = mfcc.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            logits = model(mfcc)
            loss = criterion(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


class EncoderClassifierPipeline(nn.Module):
    """
    Wraps Encoder + EmbeddingClassifier into one model so we can use
    the same training loop for both A1-A4 and A5.
    """
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(self.encoder(x))


def train(model, train_loader, val_loader, device, label):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining {label}")
    print(f"{'Epoch':<8} {'Train Loss':<14} {'Train Acc':<13} {'Val Loss':<12} {'Val Acc'}")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        val_loss, val_acc     = run_epoch(model, val_loader,   optimizer, criterion, device, train=False)
        scheduler.step()

        print(f"{epoch:<8} {train_loss:<14.4f} {train_acc:<13.4f} {val_loss:<12.4f} {val_acc:.4f}")

    return model


def main(embedding_dim):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    mfcc_transform = get_mfcc_transform()
    train_ds = SpeechCommandsDataset("training",   mfcc_transform)
    val_ds   = SpeechCommandsDataset("validation", mfcc_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # --- A1-A4: shared KWS model ---
    kws_model = KWSModel(n_mfcc=40, n_classes=35).to(device)
    train(kws_model, train_loader, val_loader, device, label="KWSModel (A1-A4)")
    torch.save(kws_model.state_dict(), os.path.join(CHECKPOINT_DIR, "kws_model.pt"))
    print(f"Saved checkpoints/kws_model.pt")

    # --- A5: encoder + classifier trained jointly ---
    encoder    = Encoder(embedding_dim=embedding_dim).to(device)
    classifier = EmbeddingClassifier(embedding_dim=embedding_dim, n_classes=35).to(device)
    pipeline   = EncoderClassifierPipeline(encoder, classifier)

    train(pipeline, train_loader, val_loader, device, label=f"Encoder+Classifier (A5, dim={embedding_dim})")
    torch.save(encoder.state_dict(),    os.path.join(CHECKPOINT_DIR, f"encoder_{embedding_dim}.pt"))
    torch.save(classifier.state_dict(), os.path.join(CHECKPOINT_DIR, f"embedding_classifier_{embedding_dim}.pt"))
    print(f"Saved checkpoints/encoder_{embedding_dim}.pt + embedding_classifier_{embedding_dim}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dim", type=int, default=64,
                        help="Embedding size for A5 encoder (default: 64)")
    args = parser.parse_args()
    main(args.embedding_dim)
