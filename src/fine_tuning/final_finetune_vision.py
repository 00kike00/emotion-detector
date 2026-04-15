import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
import json
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score, classification_report

# 1. SETUP & IMPORTS
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.config import DEVICE, PROCESSED_DIR, MODELS_DIR, CHECKPOINTS_DIR
from src.architectures.vision_net import VisionNet 

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# No augmentation for val/test — deterministic
eval_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# ── Dataset adapted to faces ──────────────────────────────────────────
class MELDFrameDataset(Dataset):
    def __init__(self, split_name: str, augment=False):
        path = PROCESSED_DIR / f"meld_{split_name}_faces.pt"
        print(f"Loading {path}...")
        self.data = torch.load(path)
        self.transform = train_transform if augment else eval_transform
        
        self.all_frames = []
        for item in self.data:
            # item['frames'] es [T, 1, 48, 48]
            frames = item['frames']
            label = item['label']
            for i in range(frames.size(0)):
                self.all_frames.append((frames[i], label))
        
        print(f"Split {split_name}: {len(self.all_frames)} individual frames loaded.")

    def __len__(self):
        return len(self.all_frames)

    def __getitem__(self, idx):
        face, label = self.all_frames[idx]
        face = self.transform(face.squeeze(0))
        return face, torch.tensor(label, dtype=torch.long)
    
    def get_labels(self):
        """Returns all labels — used for WeightedRandomSampler."""
        return [label for _, label in self.all_frames]

# ── Fine-Tuning Function ──────────────────────────────────────────────────────
def run_vision_finetuning():
    # LOAD DATA
    train_ds = MELDFrameDataset("train", augment=True)
    dev_ds = MELDFrameDataset("dev", augment=False)
    
    train_labels  = train_ds.get_labels()
    class_counts  = np.bincount(train_labels)
    class_weights = 1.0 / class_counts.astype(np.float32)
    sample_weights = torch.tensor(class_weights[train_labels], dtype=torch.float32)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    # Big batch size since we're only fine-tuning a small number of epochs
    train_loader = DataLoader(train_ds, batch_size=256, sampler=sampler, num_workers=8)
    dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, num_workers=4)

    # LOAD PRETRAINED MODEL
    with open(CHECKPOINTS_DIR / "final_vision_apso_results.json", "r") as f:
        best_params = json.load(f)
    
    model = VisionNet(
        dropout_rate=best_params['best_dropout'], 
        hidden_units=best_params['best_hidden_units']
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODELS_DIR / "final_vision_expert_best.pth", map_location=DEVICE))
    
    # OPTIMIZER (Very low learning rate for fine-tuning)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0

    EMOTION_NAMES = ['neutral', 'happiness', 'surprise', 
            'sadness', 'anger', 'disgust', 'fear']
    # TRAINING LOOP
    for epoch in range(5): 
        model.train()
        train_loss = 0
        for faces, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            faces, labels = faces.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(faces)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation 
        model.eval()
        val_loss = 0
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for faces, labels in dev_loader:
                faces, labels = faces.to(DEVICE), labels.to(DEVICE)
                outputs = model(faces)
                val_loss += nn.CrossEntropyLoss()(outputs, labels).item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_train  = train_loss / len(train_loader)
        avg_val    = val_loss   / len(dev_loader)
        val_f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train:.4f} | "
            f"Val Loss: {avg_val:.4f} | Val Macro F1: {val_f1:.4f}")

        # Save based on macro F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {'state_dict': model.state_dict()},
                MODELS_DIR / "final_vision_expert_best_ft.pth"
            )
            print(f">>> Saved: final_vision_expert_best_ft.pth (Macro F1: {val_f1:.4f})")

        # See Finetune Progress each Epoch
        print(classification_report(
            all_labels, all_preds,
            target_names=EMOTION_NAMES,
            zero_division=0
        ))
if __name__ == "__main__":
    run_vision_finetuning()