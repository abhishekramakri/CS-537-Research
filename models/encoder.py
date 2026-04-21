import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Lightweight encoder that runs on the device side for A5.
    Compresses an MFCC tensor down to a low-dimensional embedding vector.
    The embedding_dim is the bandwidth knob — smaller = fewer bytes transmitted.
    Input shape: (batch, 1, n_mfcc, time_frames)
    Output shape: (batch, embedding_dim)
    """
    def __init__(self, embedding_dim=64):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.project = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, x):
        return self.project(self.conv(x))


class EmbeddingClassifier(nn.Module):
    """
    Runs on the server side for A5.
    Takes the embedding from the device and outputs class logits.
    Trained jointly with Encoder so the embedding is task-optimized,
    not a general audio representation.
    """
    def __init__(self, embedding_dim=64, n_classes=35):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, n_classes),
        )

    def forward(self, z):
        return self.net(z)
