import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import sys
from pathlib import Path

# 1. SETUP & IMPORTS
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.config import DEVICE, PROCESSED_DIR, MODELS_DIR, CHECKPOINTS_DIR
from src.architectures.vision_net import VisionNet 

# ── Dataset adapted to faces ──────────────────────────────────────────
class MELDFrameDataset(Dataset):
    def __init__(self, split_name: str):
        path = PROCESSED_DIR / f"meld_{split_name}_faces.pt"
        print(f"Loading {path}...")
        self.data = torch.load(path)
        
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
        return face, torch.tensor(label, dtype=torch.long)

# ── Fine-Tuning Function ──────────────────────────────────────────────────────
def run_vision_finetuning():
    # 2. LOAD DATA
    train_ds = MELDFrameDataset("train")
    dev_ds = MELDFrameDataset("dev")
    
    # Big batch size since we're only fine-tuning a small number of epochs
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=8)
    dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, num_workers=4)

    # 3. LOAD PRETRAINED MODEL
    with open(CHECKPOINTS_DIR / "vision_apso_acc_results.json", "r") as f:
        best_params = json.load(f)
    
    model = VisionNet(
        dropout_rate=best_params['best_dropout'], 
        hidden_units=best_params['best_hidden_units']
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODELS_DIR / "vision_expert_best_acc.pth", map_location=DEVICE))
    
    # 4. OPTIMIZER (Very low learning rate for fine-tuning)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')

    # 5. TRAINING LOOP
    for epoch in range(5): # 5 epochs suffice for fine-tuning
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

        # Validation simple
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for faces, labels in dev_loader:
                faces, labels = faces.to(DEVICE), labels.to(DEVICE)
                outputs = model(faces)
                val_loss += criterion(outputs, labels).item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(dev_loader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        # Save best model based on validation loss
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({'state_dict': model.state_dict()}, MODELS_DIR / "vision_expert_meld_finetuned.pth")
            print(">>> Saved: vision_expert_meld_finetuned.pth")

if __name__ == "__main__":
    run_vision_finetuning()