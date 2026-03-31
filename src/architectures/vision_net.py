import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionNet(nn.Module):
    def __init__(self, dropout_rate=0.5, hidden_units=512, num_classes=7):
        """
        Custom Deep CNN for FER2013 Emotion Recognition.
        Optimized for APSO hyperparameter tuning.
        """
        super(VisionNet, self).__init__()
        
        # Block 1: Input 48x48
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Result: 24x24
        
        # Block 2: 24x24
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Result: 128x12x12
        
        # Block 3: 12x12
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Result: 256x6x6
        
        # Fully Connected Layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 6 * 6, hidden_units)
        self.bn_fc = nn.BatchNorm1d(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_units, num_classes)

    def forward(self, x, return_features=False):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(x)
        
        # Classification
        x = self.flatten(x)
        
        # Feature Extraction Point -> These features can be used by the Manager for decision making
        features = F.relu(self.bn_fc(self.fc1(x))) 
        
        x = self.dropout(features)
        logits = self.fc2(x)
        
        if return_features:
            return logits, features # Return both for training and Manager
        return logits # For training