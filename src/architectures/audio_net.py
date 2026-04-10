import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioNet(nn.Module):
    def __init__(self, dropout_rate=0.3, hidden_units=256, num_classes=7):
        """
        CNN for RAVDESS Speech Emotion Recognition.
        Designed for audio spectrogram input of shape [1, 128, 172].
        """
        super(AudioNet, self).__init__()

        # Block 1: [1, 128, 172] -> [32, 64, 86]
        self.conv1 = nn.Conv2d(1,  32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(dropout_rate * 0.5)

        # Block 2: [32, 32, 64] -> [64, 16, 32]
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(dropout_rate * 0.5)

        # Block 3: [64, 16, 32] -> [128, 8, 16]
        self.conv5 = nn.Conv2d(64,  128, kernel_size=3, padding=1)
        self.bn5   = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6   = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout2d(dropout_rate * 0.5)

        # Block 4: [128, 8, 16] -> [256, 4, 8]
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7   = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8   = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout2d(dropout_rate * 0.5)

        # Global Average Pooling — collapses [256, 4, 8] -> [256]
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Fully Connected
        self.flatten  = nn.Flatten()
        self.fc1      = nn.Linear(256, hidden_units)
        self.bn_fc    = nn.BatchNorm1d(hidden_units)
        self.dropout  = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_units, num_classes)

    def forward(self, x, return_features=False):
        # Block 1
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = F.elu(self.bn3(self.conv3(x)))
        x = F.elu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = F.elu(self.bn5(self.conv5(x)))
        x = F.elu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        # Block 4
        x = F.elu(self.bn7(self.conv7(x)))
        x = F.elu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        x = self.drop4(x)

        # Global average pool + flatten -> [B, 256]
        x = self.gap(x)
        x = self.flatten(x)

        # Feature extraction point — Could be used by Manager for fusion
        features = F.elu(self.bn_fc(self.fc1(x)))

        x      = self.dropout(features)
        logits = self.fc2(x)

        if return_features:
            return logits, features
        return logits