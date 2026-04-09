import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# 1. PATH SETUP
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.config import GOEMOTIONS_DIR

def run_goemotions_eda():
    # File Paths
    mapping_path = GOEMOTIONS_DIR / "ekman_mapping.json"
    emotions_txt = GOEMOTIONS_DIR / "emotions.txt"
    splits = ['train.tsv', 'dev.tsv', 'test.tsv']

    # 2. LOAD MAPPING LOGIC (Mirroring your Dataset class)
    if not emotions_txt.exists() or not mapping_path.exists():
        print(f"Error: Required files not found in {GOEMOTIONS_DIR}")
        return

    with open(emotions_txt, 'r') as f:
        idx_to_go = [line.strip() for line in f if line.strip()]

    with open(mapping_path, 'r') as f:
        mapping_json = json.load(f)

    # Invert mapping: {7-class: [go_list]} -> {go_name: 7-class}
    go_to_7 = {}
    for seven_class, go_list in mapping_json.items():
        for go_name in go_list:
            go_to_7[go_name] = seven_class
    if 'neutral' not in go_to_7: 
        go_to_7['neutral'] = 'neutral'

    # Create a lookup for indices to the final 7 classes (Capitalized)
    idx_to_seven = {}
    for i, name in enumerate(idx_to_go):
        mapped = go_to_7.get(name, 'neutral')
        idx_to_seven[i] = mapped.capitalize()

    # 3. PROCESS DATA
    all_data = []
    split_summaries = {}

    for split in splits:
        path = GOEMOTIONS_DIR / split
        if not path.exists():
            print(f"Warning: {split} not found. Skipping.")
            continue

        # Load Raw TSV
        df_raw = pd.read_csv(path, sep='\t', header=None, names=['text', 'labels', 'id'])
        
        # Apply your Dataset logic: Take the FIRST label index
        # We only keep rows that map to our target 7 classes (which 'neutral' is part of)
        df_raw['first_label_idx'] = df_raw['labels'].apply(lambda x: int(str(x).split(',')[0]))
        df_raw['Mapped_Emotion'] = df_raw['first_label_idx'].map(idx_to_seven)
        
        # Keep track of split
        df_raw['Split'] = split
        all_data.append(df_raw)
        
        # Calculate split-specific stats
        split_summaries[split] = {
            'count': len(df_raw),
            'proportions': df_raw['Mapped_Emotion'].value_counts(normalize=True) * 100,
            'class_counts': df_raw['Mapped_Emotion'].value_counts()
        }

    # Combine for global analysis
    full_df = pd.concat(all_data)

    # 4. PRINT GLOBAL SUMMARY (COUNTS & TOTALS)
    print("\n" + "="*60)
    print("           GOEMOTIONS DATASET: GLOBAL SUMMARY")
    print("="*60)
    
    grand_total = 0
    for split, info in split_summaries.items():
        print(f"{split.upper():12} | Samples: {info['count']:,}")
        grand_total += info['count']
    
    print("-" * 60)
    print(f"{'GRAND TOTAL':12} | Samples: {grand_total:,}")
    print("="*60)

    # 5. PRINT CLASS BREAKDOWN (COUNTS & %)
    # Create DataFrames for side-by-side comparison
    counts_df = pd.DataFrame({k: v['class_counts'] for k, v in split_summaries.items()}).fillna(0).astype(int)
    pct_df = pd.DataFrame({k: v['proportions'] for k, v in split_summaries.items()}).fillna(0)

    print("\n--- SAMPLES PER EMOTION (ABSOLUTE COUNTS) ---")
    print(counts_df.sort_values(by='train.tsv', ascending=False))

    print("\n--- CLASS DISTRIBUTION (PERCENTAGES %) ---")
    print(pct_df.sort_values(by='train.tsv', ascending=False).round(2))
    print("="*60)

    # 6. VISUALIZATION
    plt.figure(figsize=(14, 7))
    train_df = full_df[full_df['Split'] == 'train.tsv']
    
    # Calculate order based on frequency
    order = train_df['Mapped_Emotion'].value_counts().index

    sns.countplot(data=train_df, x='Mapped_Emotion', order=order, palette='viridis')
    
    plt.title('GoEmotions Class Distribution (Mapped to 7 Classes - Training Set)')
    plt.ylabel('Number of Samples')
    plt.xlabel('Emotion')
    plt.xticks(rotation=45)

    # Annotate bars with counts
    ax = plt.gca()
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='center', 
                    fontsize=10, color='black', xytext=(0, 7), 
                    textcoords='offset points')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_goemotions_eda()