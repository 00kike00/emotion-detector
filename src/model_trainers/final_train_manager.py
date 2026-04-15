import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import random
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report

# Path Fix
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.config import DEVICE, CHECKPOINTS_DIR, MODELS_DIR, PLOTS_DIR
from src.architectures.manager_net import ManagerNet
from src.data_pipeline.loaders import get_meld_loaders

SEED = 42

EMOTION_NAMES = ['neutral', 'happiness', 'surprise',
                 'sadness', 'anger', 'disgust', 'fear']

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
    with open(CHECKPOINTS_DIR / "final_manager_apso_results.json", "r") as f:
        best_params = json.load(f)

    print(f"--- Training Final Manager Network ---")
    print(f"Parameters: {best_params}")

    history = {
        'train_loss': [],
        'val_f1':        [],
        'val_acc':       [],
        'val_precision': [],
        'val_recall':    []
    }

    # 2. Setup Data & Model
    train_loader, dev_loader, test_loader = get_meld_loaders()

    model = ManagerNet(
        num_classes=7,
        vision_input_dim=7,
        text_input_dim=7,
        hidden_dim=best_params['best_hidden_units'],
        num_layers=best_params['best_num_layers'],
        dropout_rate=best_params['best_dropout'],
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=10 ** best_params['best_learning_rate'])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=8, factor=0.4
    )

    # 3. Training Loop
    epochs               = 1000
    best_f1              = 0.0
    patience_counter     = 0
    early_stop_patience  = 50

    for epoch in range(epochs):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for vision_seq, lengths, text_logits, labels in train_loader:
            vision_seq  = vision_seq.to(DEVICE)   # [B, T_max, 7]
            text_logits = text_logits.to(DEVICE)   # [B, 7]
            labels      = labels.to(DEVICE)        # [B]

            optimizer.zero_grad()
            outputs = model(vision_seq, text_logits)  # [B, 7]
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for vision_seq, lengths, text_logits, labels in dev_loader:
                vision_seq  = vision_seq.to(DEVICE)
                text_logits = text_logits.to(DEVICE)
                labels      = labels.to(DEVICE)

                outputs = model(vision_seq, text_logits)
                preds   = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_f1        = f1_score(all_labels, all_preds, average='macro',
                                  zero_division=0) * 100
        val_acc       = accuracy_score(all_labels, all_preds) * 100
        val_precision = precision_score(all_labels, all_preds, average='macro',
                                         zero_division=0) * 100
        val_recall    = recall_score(all_labels, all_preds, average='macro',
                                      zero_division=0) * 100

        history['train_loss'].append(train_loss / len(train_loader))
        history['val_f1'].append(val_f1)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)

        scheduler.step(val_f1)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {train_loss/len(train_loader):.4f} | "
              f"Val Macro F1: {val_f1:.2f}%")

        # Save Best Model
        if val_f1 > best_f1:
            best_f1          = val_f1
            patience_counter = 0
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'config':     best_params
                },
                MODELS_DIR / "final_manager_best.pth"
            )
            print(f"  [SAVED] New best Macro F1: {best_f1:.2f}%")
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

    # 4. Final Test Evaluation
    print("\n--- Running Final Test Evaluation ---")
    checkpoint = torch.load(MODELS_DIR / "final_manager_best.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_preds  = []
    test_labels = []

    with torch.no_grad():
        for vision_seq, lengths, text_logits, labels in test_loader:
            vision_seq  = vision_seq.to(DEVICE)
            text_logits = text_logits.to(DEVICE)
            labels      = labels.to(DEVICE)

            outputs = model(vision_seq, text_logits)
            preds   = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    final_test_f1 = f1_score(test_labels, test_preds, average='macro',
                              zero_division=0) * 100
    print(f"Final Test Macro F1: {final_test_f1:.2f}%")
    print(classification_report(
        test_labels, test_preds,
        target_names=EMOTION_NAMES,
        zero_division=0
    ))

    # 5. Generate and Save Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Loss', color='royalblue')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_f1'],        label='Val F1 Macro',  color='seagreen')
    plt.plot(history['val_acc'],       label='Val Accuracy',   color='orange')
    plt.plot(history['val_precision'], label='Val Precision',  color='purple')
    plt.plot(history['val_recall'],    label='Val Recall',     color='brown')
    plt.axhline(y=final_test_f1, color='r', linestyle='--',
                label=f'Test F1: {final_test_f1:.1f}%')
    plt.title('Metrics Over Epochs')
    plt.xlabel('Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "final_manager_training_curves.png")
    plt.show()

    print(f"\nFinal Training Complete. Best Val Macro F1: {best_f1:.2f}%")


if __name__ == "__main__":
    train_final()