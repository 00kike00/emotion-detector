import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from transformers import get_linear_schedule_with_warmup
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

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

def train_final():
    # 1. Load APSO Results
    with open(CHECKPOINTS_DIR / "text_apso_results.json", "r") as f:
        best_params = json.load(f)
    
    print(f"--- Training Final Text Expert ---")
    print(f"Parameters: {best_params}")

    history = {'train_loss': [], 'val_acc': []}

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
    print("RoBERTa frozen for warmup phase.")

    optimizer = optim.AdamW([
        {'params': model.roberta.parameters(),    'lr': 1e-5},
        {'params': model.lstm.parameters(),       'lr': best_params['best_learning_rate']},
        {'params': model.classifier.parameters(), 'lr': best_params['best_learning_rate']},
    ], weight_decay=0.01)

    criterion = nn.CrossEntropyLoss()

    epochs = 15
    early_stop_patience = 5
    UNFREEZE_EPOCH = 3

    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = num_training_steps // 10

    from transformers import get_linear_schedule_with_warmup
    warmup_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    scaler = GradScaler('cuda')

    # 3. Training Loop
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):

        # Unfreeze RoBERTa after warmup epochs
        if epoch == UNFREEZE_EPOCH:
            print("Unfreezing RoBERTa for full fine-tuning...")
            for param in model.roberta.parameters():
                param.requires_grad = True

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
            warmup_scheduler.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # Validation Phase
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in valid_loader:
                ids, mask, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['label'].to(DEVICE)
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(ids, mask)
                _, predicted = torch.max(outputs, 1)
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc  = 100 * correct / total
        avg_loss = train_loss / len(train_loader)
        
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)

        plateau_scheduler.step(val_acc)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'state_dict': model.state_dict(),
                'config': best_params
            }, MODELS_DIR / "text_expert_best.pth")
            patience_counter = 0
            print(f"  [SAVED] New best accuracy: {best_acc:.2f}%")
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

    # 4. Final Test Evaluation
    print("\n--- Running Final Test Evaluation ---")
    checkpoint = torch.load(MODELS_DIR / "text_expert_best.pth", weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            ids, mask, labels = batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE), batch['label'].to(DEVICE)
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(ids, mask)
            _, predicted = torch.max(outputs, 1)
            test_total   += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    final_test_acc = 100 * test_correct / test_total
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")

    # 5. Generate and Save Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Loss', color='darkorange')
    plt.title('Text Training Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Acc', color='crimson')
    plt.axhline(y=final_test_acc, color='blue', linestyle='--', label=f'Test: {final_test_acc:.1f}%')
    plt.title('Text Validation Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "text_training_curves.png")
    plt.show()

if __name__ == "__main__":
    train_final()