import torch
import torch.nn as nn


class KWSModel(nn.Module):
    """
    Compact CNN for keyword spotting, loosely based on Zhang et al. 2017.
    Expects input shape: (batch, 1, n_mfcc, time_frames)
    e.g. (batch, 1, 40, 101) for 40 MFCCs over a 1-second clip at 10ms hop.
    """
    def __init__(self, n_mfcc=40, n_classes=35):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # global average pool — collapses spatial dims to 1x1, works on MPS
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.conv(x))
