import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# ── Setup Paths ──────────────────────────────────────────────────────────────
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.data_pipeline.loaders import MELDLogitsDataset

def run_dataset_eda():
    mapping = {
        0: 'Neutral', 1: 'Happiness', 2: 'Surprise', 
        3: 'Sadness', 4: 'Anger', 5: 'Disgust', 6: 'Fear'
    }

    print("--- Analyzing Dataset ---")
    
    splits = ["train", "dev", "test"]
    results = []

    for name in splits:
        ds = MELDLogitsDataset(name)
        
        # Extract labels and lengths for EDA
        labels = [s['label'] for s in ds.samples]
        lengths = [s['vision_seq'].shape[0] for s in ds.samples]
        
        df_split = pd.DataFrame({
            'label': labels,
            'v_len': lengths,
            'Split': name
        })
        df_split['Emotion'] = df_split['label'].map(mapping)
        results.append(df_split)

    df = pd.concat(results, ignore_index=True)

    # ── Visualization ──
    plt.figure(figsize=(12, 5))

    # Distribution of classes in Train set (Post-filtering)
    plt.subplot(1, 2, 1)
    sns.countplot(data=df[df['Split']=='train'], x='Emotion', 
                  order=[mapping[i] for i in range(7)], palette='viridis')
    plt.title('Final distribution in Train (Post-filtering)')
    plt.xticks(rotation=45)

    # Lengths of video sequences by emotion
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x='Emotion', y='v_len')
    plt.title('Frames of Video by Emotion')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    print("\nClass distribution in Train (Post-filtering):")
    print(df[df['Split']=='train']['Emotion'].value_counts().rename(index=mapping)/len(df[df['Split']=='train'])*100)
    
    # Show exact counts in Train set
    counts = df[df['Split']=='train']['label'].value_counts().sort_index()
    print("\nExact counts in Train:")
    print(counts.rename(index=mapping))

if __name__ == "__main__":
    run_dataset_eda()