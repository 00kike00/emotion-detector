import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Path Setup
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.config import FER_DIR

def run_fer_eda():
    csv_path = FER_DIR / "fer2013new.csv"
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found.")
        return

    # 1. Load Data
    df = pd.read_csv(csv_path)

    # 2. Define the columns we care about
    # Note: We are ignoring 'contempt', 'unknown', and 'NF'
    emotion_cols = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

    # 3. Get the Max Vote
    # This finds the column name with the highest number of votes for each row
    df['Mapped_Emotion'] = df[emotion_cols].idxmax(axis=1)

    # 4. Filter by Usage (Train vs Test/PublicTest)
    # FER2013 uses 'Training', 'PublicTest', and 'PrivateTest'
    train_df = df[df['Usage'] == 'Training']
    test_df = df[df['Usage'].isin(['PublicTest', 'PrivateTest'])]

    # 5. Print Basic Stats
    print(f"--- FER+ Dataset Summary ---")
    print(f"Total Images: {len(df)}")
    print(f"Train Samples: {len(train_df)}")
    print(f"Test Samples:  {len(test_df)}")
    print("-" * 30)

    # 6. Calculate Proportions
    train_counts = train_df['Mapped_Emotion'].str.capitalize().value_counts(normalize=True) * 100
    test_counts = test_df['Mapped_Emotion'].str.capitalize().value_counts(normalize=True) * 100

    eda_df = pd.DataFrame({
        'Train %': train_counts,
        'Test %': test_counts
    }).sort_values(by='Train %', ascending=False)

    print("\nClass Proportions (%):")
    print(eda_df.round(2))

    # 7. Visualize
    plt.figure(figsize=(12, 6))
    
    # We'll plot the training distribution
    sns.countplot(data=train_df, x='Mapped_Emotion', 
                  order=train_df['Mapped_Emotion'].value_counts().index, 
                  palette='magma')
    
    plt.title('Emotion Distribution in FER+ (Train Set)')
    plt.ylabel('Number of Images')
    plt.xlabel('Emotion')
    plt.xticks(rotation=45)
    
    # Add counts on top
    ax = plt.gca()
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5), 
                    textcoords='offset points')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_fer_eda()