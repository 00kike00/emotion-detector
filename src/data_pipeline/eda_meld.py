import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.config import MELD_DIR

def run_meld_eda():
    # 1. Load Datasets
    try:
        train_df = pd.read_csv(MELD_DIR / "train" / "train_sent_emo.csv", encoding='latin-1')
        dev_df = pd.read_csv(MELD_DIR / "dev" / "dev_sent_emo.csv", encoding='latin-1')
    except FileNotFoundError:
        print("Error: CSV files not found. Check your MELD_DIR path.")
        return

    # 2. Map Labels my Classes
    mapping = {
        'neutral': 'Neutral', 'joy': 'Happiness', 'surprise': 'Surprise', 
        'sadness': 'Sadness', 'anger': 'Anger', 'disgust': 'Disgust', 'fear': 'Fear'
    }
    
    train_df['Mapped_Emotion'] = train_df['Emotion'].map(mapping)
    dev_df['Mapped_Emotion'] = dev_df['Emotion'].map(mapping)

    # 3. Print Basic Stats
    print(f"--- MELD Dataset Summary ---")
    print(f"Train Samples: {len(train_df)}")
    print(f"Dev Samples:   {len(dev_df)}")
    print(f"Total:         {len(train_df) + len(dev_df)}")
    print("-" * 30)

    # 4. Calculate Proportions
    train_counts = train_df['Mapped_Emotion'].value_counts(normalize=True) * 100
    dev_counts = dev_df['Mapped_Emotion'].value_counts(normalize=True) * 100

    eda_df = pd.DataFrame({
        'Train %': train_counts,
        'Dev %': dev_counts
    }).sort_values(by='Train %', ascending=False)

    print("\nClass Proportions (%):")
    print(eda_df.round(2))

    # 5. Visualize
    plt.figure(figsize=(12, 6))
    
    sns.countplot(data=train_df, x='Mapped_Emotion', 
                  order=train_df['Mapped_Emotion'].value_counts().index, 
                  palette='viridis')
    
    plt.title('Emotion Distribution in MELD (Train Set)')
    plt.ylabel('Number of Samples')
    plt.xlabel('Emotion')
    plt.xticks(rotation=45)
    
    # Add counts on top of bars
    ax = plt.gca()
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5), 
                    textcoords='offset points')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_meld_eda()