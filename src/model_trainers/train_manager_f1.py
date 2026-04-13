import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import random
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Path Fix
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.config import DEVICE, CHECKPOINTS_DIR, MODELS_DIR, PLOTS_DIR
from src.architectures.manager_net import ManagerNet
from src.data_pipeline.loaders import get_meld_loaders

SEED = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_final():
    set_seed(SEED)

    # 1. Load APSO Results
    with open(CHECKPOINTS_DIR / "manager_apso_f1_results.json", "r") as f:
        best_params = json.load(f)
    
    print(f"--- Training Final Manager ---")
    print(f"Parameters: {best_params}")

    history = {
        'train_loss': [],
        'val_f1': []
    }

    # 2. Setup Data & Model
    train_loader, valid_loader, test_loader = get_meld_loaders()
    model = ManagerNet(
        dropout_rate=best_params['best_dropout'], 
        hidden_dim=best_params['best_hidden_dim'],
        num_layers=best_params['best_num_layers']
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['best_learning_rate'])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)

    # 3. Training Loop
    epochs = 100
    best_f1 = 0.0
    patience_counter = 0
    early_stop_patience = 20

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for vision_seq, lengths, text_logits, labels in train_loader:
            vision_seq  = vision_seq.to(DEVICE)
            text_logits = text_logits.to(DEVICE)
            labels      = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(vision_seq, text_logits)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        # Validation
        model.eval()
        all_preds  = []
        all_labels = []
        with torch.no_grad():
            for vision_seq, lengths, text_logits, labels in valid_loader:
                vision_seq  = vision_seq.to(DEVICE)
                text_logits = text_logits.to(DEVICE)

                outputs = model(vision_seq, text_logits)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        val_f1 = f1_score(all_labels, all_preds, average='macro') * 100

        history['train_loss'].append(train_loss / len(train_loader))
        history['val_f1'].append(val_f1)

        scheduler.step(val_f1)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val F1 Macro: {val_f1:.2f}%")

        # Save Best Model
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), MODELS_DIR / "manager_expert_best_f1.pth")
            patience_counter = 0
            print(f"  [SAVED] New best F1 Macro: {best_f1:.2f}%")
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

    # 4. Final Test Evaluation
    print("\n--- Running Final Test Evaluation ---")
    model.load_state_dict(torch.load(MODELS_DIR / "manager_expert_best_f1.pth"))
    model.eval()
    
    test_preds  = []
    test_labels = []
    with torch.no_grad():
        for vision_seq, lengths, text_logits, labels in test_loader:
            vision_seq  = vision_seq.to(DEVICE)
            text_logits = text_logits.to(DEVICE)
            labels      = labels.to(DEVICE)

            outputs = model(vision_seq, text_logits)
            _, predicted = torch.max(outputs.data, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    final_test_f1 = f1_score(test_labels, test_preds, average='macro') * 100
    print(f"Final Test F1 Macro: {final_test_f1:.2f}%")

    # 5. Generate and Save Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Loss', color='royalblue')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_f1'], label='Val F1 Macro', color='seagreen')
    plt.axhline(y=final_test_f1, color='r', linestyle='--', label=f'Test F1: {final_test_f1:.1f}%')
    plt.title('Validation F1 Macro')
    plt.xlabel('Epochs')
    plt.legend()
    print(f"Final Training Complete. Best Val F1 Macro: {best_f1:.2f}%")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "manager_training_f1_curves.png")
    plt.show()

if __name__ == "__main__":
    train_final()