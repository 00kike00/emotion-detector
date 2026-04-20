import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# 1. PATH SETUP
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.config import PROCESSED_DIR, PLOTS_DIR

def evaluate_experts_on_meld(split_name="test"):
    """
    Evaluates how the Vision and Text experts perform individually on the MELD dataset
    using the pre-extracted logits.
    """
    print(f"--- Evaluating Experts Independently on MELD ({split_name.upper()}) ---")
    
    # 2. LOAD PRE-EXTRACTED LOGITS
    # These are the files generated with the feature_extractor scripts
    v_path = PROCESSED_DIR / f"final_meld_{split_name}_vision_logits.pt"
    t_path = PROCESSED_DIR / f"final_meld_{split_name}_text_logits.pt"
    
    if not v_path.exists() or not t_path.exists():
        print(f"Error: Logit files not found in {PROCESSED_DIR}")
        return

    vision_data = torch.load(v_path)
    text_data = torch.load(t_path)
    
    # 3. ALIGN DATA
    # Create lookups to ensure we compare the same utterances
    text_lookup = {s['id']: s for s in text_data}
    class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
    
    results = {
        'vision': {'preds': [], 'labels': []},
        'text':   {'preds': [], 'labels': []}
    }

    for v_sample in vision_data:
        uid = v_sample['id']
        if uid not in text_lookup:
            continue
            
        t_sample = text_lookup[uid]
        label = v_sample['label']

        # VISION PREDICTION:
        # v_sample['vision_logits'] is [T, 7]. We take the mean over time (T)
        v_logits_mean = v_sample['vision_logits'].mean(dim=0)
        v_pred = torch.argmax(v_logits_mean).item()
        
        # TEXT PREDICTION:
        # t_sample['text_logits'] is already [7]
        t_pred = torch.argmax(t_sample['text_logits']).item()

        # Store
        results['vision']['preds'].append(v_pred)
        results['vision']['labels'].append(label)
        results['text']['preds'].append(t_pred)
        results['text']['labels'].append(label)

    # 4. PLOTTING
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    for i, mod in enumerate(['vision', 'text']):
        y_true = results[mod]['labels']
        y_pred = results[mod]['preds']
        
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Raw CM
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues' if mod=='text' else 'Greens',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[i, 0])
        axes[i, 0].set_title(f'{mod.upper()} Expert: Raw Confusion Matrix', fontsize=14)
        
        # Normalized CM
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues' if mod=='text' else 'Greens',
                    xticklabels=class_names, yticklabels=class_names, ax=axes[i, 1])
        axes[i, 1].set_title(f'{mod.upper()} Expert: Normalized CM (%)', fontsize=14)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"final_meld_independent_experts_{split_name}_ft_question.png")
    
    # 5. PRINT REPORTS
    print("\n" + "="*30)
    print(f"VISION EXPERT ON MELD ({split_name})")
    vision_report = classification_report(results['vision']['labels'], results['vision']['preds'], target_names=class_names)
    print(vision_report)
    with open(PLOTS_DIR / f"final_meld_vision_evaluation_{split_name}.txt", "w") as f:
        f.write(vision_report)

    print("\n" + "="*30)
    print(f"TEXT EXPERT ON MELD ({split_name})")
    text_report = classification_report(results['text']['labels'], results['text']['preds'], target_names=class_names)
    print(text_report)
    with open(PLOTS_DIR / f"final_meld_text_evaluation_{split_name}.txt", "w") as f:
        f.write(text_report)

if __name__ == "__main__":
    evaluate_experts_on_meld("test") # Choose Data set