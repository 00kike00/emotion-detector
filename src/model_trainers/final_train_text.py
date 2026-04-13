import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import numpy as np

# Path Fix
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.config import DEVICE, CHECKPOINTS_DIR, MODELS_DIR, PLOTS_DIR
from src.architectures.text_net import RobertaBiLSTM
from src.data_pipeline.loaders import get_text_loaders

# Silence RoBERTa loading warnings
from transformers import logging
logging.set_verbosity_error()
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
    with open(CHECKPOINTS_DIR / "final_text_apso_results.json", "r") as f:
        best_params = json.load(f)
    
    print(f"--- Training Final Text Expert ---")
    print(f"Parameters: {best_params}")

    history = {
        'train_loss': [],
        'val_f1': [],
        'val_acc': [],
        'val_precision': [],
        'val_recall': []
    }

    # 2. Setup Data & Model
    train_loader, valid_loader, test_loader = get_text_loaders()
    
    model = RobertaBiLSTM(
        num_classes=7, 
        hidden_dim=int(best_params['best_hidden_units']), 
        dropout=best_params['best_dropout'], 
        pooling_mode=best_params['best_pooling_mode']
    ).to(DEVICE)

    # Freeze RoBERTa initially
    for param in model.roberta.parameters():
        param.requires_grad = False

    # Unfreeze last 2 layers of RoBERTa to match the APSO evaluation setup
    for param in model.roberta.encoder.layer[-2:].parameters():
        param.requires_grad = True

    lr = 10 ** best_params['best_learning_rate']
    optimizer = optim.AdamW([
        {'params': model.lstm.parameters(), 'lr': lr},
        {'params': model.classifier.parameters(), 'lr': lr},
        {'params': model.roberta.encoder.layer[10:12].parameters(), 'lr': lr},
        {'params': model.roberta.pooler.parameters(), 'lr': lr},
        
        # Earlier RoBERTa Layers (We give them a smaller LR since they were not "seen" by the APSO and to prevent catastrophic forgetting)
        {'params': model.roberta.encoder.layer[6:10].parameters(), 'lr': lr * 0.5},
        {'params': model.roberta.encoder.layer[0:6].parameters(), 'lr': lr * 0.1},
        {'params': model.roberta.embeddings.parameters(), 'lr': lr * 0.1},
    ], weight_decay=0.01)

    criterion = nn.CrossEntropyLoss()

    epochs = 15
    early_stop_patience = 5
    UNFREEZE_EPOCH = 5
    UNFREEZE_EPOCH_2 = 10

    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    scaler = GradScaler('cuda')

    # 3. Training Loop
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(epochs):

        # Unfreeze RoBERTa after warmup epochs
        if epoch == UNFREEZE_EPOCH:
            print("Unfreezing RoBERTa for more in depth fine-tuning...")
            for param in model.roberta.encoder.layer[6:10].parameters():
                param.requires_grad = True
                plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
        elif epoch == UNFREEZE_EPOCH_2:
            print("Unfreezing all RoBERTa layers for final fine-tuning...")
            for param in model.roberta.encoder.layer[0:6].parameters():
                param.requires_grad = True
                plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in loop:
            ids    = batch['input_ids'].to(DEVICE)
            mask   = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(ids, mask)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Validation Phase
        model.eval()
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for batch in valid_loader:
                ids, mask, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['label'].to(DEVICE)
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(ids, mask)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(all_labels, all_preds, average='macro') * 100
        val_acc = accuracy_score(all_labels, all_preds) * 100
        val_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100
        val_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100


        avg_loss = train_loss / len(train_loader)
        
        history['train_loss'].append(avg_loss)
        history['val_f1'].append(val_f1)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)

        plateau_scheduler.step(val_f1)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.2f}%")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'state_dict': model.state_dict(),
                'config': best_params
            }, MODELS_DIR / "final_text_expert_best.pth")
            patience_counter = 0
            print(f"  [SAVED] New best F1: {best_f1:.2f}%")
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

    # 4. Final Test Evaluation
    print("\n--- Running Final Test Evaluation ---")
    checkpoint = torch.load(MODELS_DIR / "final_text_expert_best.pth", weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    test_preds  = []
    test_labels = []

    with torch.no_grad():
        for batch in test_loader:
            ids, mask, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['label'].to(DEVICE)
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(ids, mask)
            _, predicted = torch.max(outputs, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    final_test_f1 = f1_score(test_labels, test_preds, average='macro') * 100
    print(f"Final Test F1: {final_test_f1:.2f}%")

    # 5. Generate and Save Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Loss', color='darkorange')
    plt.title('Text Training Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_f1'], label='Val F1 Macro', color='seagreen')
    plt.plot(history['val_acc'], label='Val Accuracy', color='orange')
    plt.plot(history['val_precision'], label='Val Precision', color='purple')
    plt.plot(history['val_recall'], label='Val Recall', color='brown')
    plt.axhline(y=final_test_f1, color='blue', linestyle='--', label=f'Test: {final_test_f1:.1f}%')
    plt.title('Text Validation Metrics')
    plt.xlabel('Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "final_text_training_curves.png")
    plt.show()

if __name__ == "__main__":
    train_final()