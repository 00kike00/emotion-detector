import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import json
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# 1. PATH SETUP
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.config import DEVICE, CHECKPOINTS_DIR, MODELS_DIR, PLOTS_DIR
from src.architectures.manager_net import ManagerNet
from src.data_pipeline.loaders import get_meld_loaders

def evaluate_manager():
    print(f"--- Evaluating Manager on {DEVICE} ---")
    
    # Ensure PLOTS_DIR exists
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. LOAD DATA
    _, _, test_loader = get_meld_loaders()
    
    # 3. LOAD APSO RESULTS & MODEL
    with open(CHECKPOINTS_DIR / "final_manager_apso_results.json", "r") as f:
        best_params = json.load(f)
    
    model = ManagerNet(
        dropout_rate=best_params['best_dropout'], 
        hidden_dim=best_params['best_hidden_units'],
        num_layers=best_params['best_num_layers']
    ).to(DEVICE)
    
    checkpoint = torch.load(MODELS_DIR / "final_manager_best.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 4. INFERENCE
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for vision_seq, lengths, text_logits, labels in test_loader:
            vision_seq  = vision_seq.to(DEVICE)
            text_logits = text_logits.to(DEVICE)
            labels      = labels.to(DEVICE)

            outputs = model(vision_seq, text_logits)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

    # 5. GENERATE PLOTS
    # We will create a dual-plot: Raw counts and Normalized (Percentages)
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))

    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalize by row

    # Left Plot: Raw Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax[0])
    ax[0].set_title('Vision Expert: Raw Confusion Matrix', fontsize=14)
    ax[0].set_ylabel('Actual')
    ax[0].set_xlabel('Predicted')

    # Right Plot: Normalized (%)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names, ax=ax[1])
    ax[1].set_title('Vision Expert: Normalized Confusion Matrix (%)', fontsize=14)
    ax[1].set_ylabel('Actual')
    ax[1].set_xlabel('Predicted')

    plt.tight_layout()
    
    # 6. SAVE EVERYTHING TO PLOTS_DIR
    plot_filename = PLOTS_DIR / "final_manager_evaluation.png"
    plt.savefig(plot_filename)
    
    # Also save the text report for quick reference
    report = classification_report(all_labels, all_preds, target_names=class_names)
    with open(PLOTS_DIR / "final_manager_classification_report.txt", "w") as f:
        f.write(report)

    print(f"\n--- Evaluation Successful ---")
    print(f"Figures saved to: {PLOTS_DIR}")
    print(report)

if __name__ == "__main__":
    evaluate_manager()