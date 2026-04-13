import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from torch.amp.autocast_mode import autocast

# 1. PATH SETUP
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.config import DEVICE, MODELS_DIR, PLOTS_DIR
from src.architectures.text_net import RobertaBiLSTM
from src.data_pipeline.loaders import get_text_loaders

from transformers import logging
logging.set_verbosity_error()

def evaluate_text_expert():
    print(f"--- Evaluating Text Expert on {DEVICE} ---")
    
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. LOAD DATA
    _, _, test_loader = get_text_loaders()
    
    # 3. LOAD CHECKPOINT & MODEL
    checkpoint = torch.load(MODELS_DIR / "final_text_expert_best.pth", 
                            map_location=DEVICE, weights_only=False)
    best_params = checkpoint['config']

    model = RobertaBiLSTM(
        num_classes=7,
        hidden_dim=int(best_params['best_hidden_units']),
        dropout=best_params['best_dropout'],
        pooling_mode=best_params['best_pooling_mode']
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 4. INFERENCE
    all_preds  = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            ids    = batch['input_ids'].to(DEVICE)
            mask   = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(ids, mask)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

    # 5. GENERATE PLOTS
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    cm      = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax[0])
    ax[0].set_title('Text Expert: Raw Confusion Matrix', fontsize=14)
    ax[0].set_ylabel('Actual')
    ax[0].set_xlabel('Predicted')

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names, ax=ax[1])
    ax[1].set_title('Text Expert: Normalized Confusion Matrix (%)', fontsize=14)
    ax[1].set_ylabel('Actual')
    ax[1].set_xlabel('Predicted')

    plt.tight_layout()

    # 6. SAVE
    plt.savefig(PLOTS_DIR / "final_text_evaluation.png")

    report = classification_report(all_labels, all_preds, target_names=class_names)
    with open(PLOTS_DIR / "final_text_classification_report.txt", "w") as f:
        f.write(report)

    print(f"\n--- Evaluation Successful ---")
    print(f"Figures saved to: {PLOTS_DIR}")
    print(report)

if __name__ == "__main__":
    evaluate_text_expert()