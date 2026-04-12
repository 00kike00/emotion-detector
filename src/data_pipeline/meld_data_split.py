import pandas as pd
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.config import MELD_DIR, PROCESSED_DIR

SEED = 42
TEST_SIZE = 1000

TRAIN_PROPORTIONS = {
    'neutral':   0.4715,
    'happiness': 0.1745,
    'surprise':  0.1206,
    'anger':     0.1110,
    'sadness':   0.0684,
    'disgust':   0.0271,
    'fear':      0.0268,
}

def prepare_meld_csvs():
    # Load and clean
    df = pd.read_csv(MELD_DIR / "train/train_sent_emo.csv", encoding='latin-1')
    df['Emotion']   = df['Emotion'].replace({'joy': 'happiness'})
    df['Utterance'] = df['Utterance'].str.replace('\u2019', "'", regex=False)

    # Stratified split using fixed proportions
    test_rows = []
    for emotion, proportion in TRAIN_PROPORTIONS.items():
        n_samples  = round(TEST_SIZE * proportion)
        class_rows = df[df['Emotion'] == emotion]

        if len(class_rows) < n_samples:
            raise ValueError(
                f"Not enough '{emotion}' samples: need {n_samples}, have {len(class_rows)}"
            )

        test_rows.append(class_rows.sample(n=n_samples, random_state=SEED))

    test_df  = pd.concat(test_rows)
    train_df = df.drop(index=test_df.index)

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "meld_train_split.csv", index=False)
    test_df.to_csv(PROCESSED_DIR  / "meld_test_split.csv",  index=False)

    # Validation printout
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")
    print("\nTest class distribution:")
    dist = test_df['Emotion'].value_counts()
    for emotion, count in dist.items():
        print(f"  {emotion:<12} {count:>4}  ({count/len(test_df)*100:.1f}%)")

if __name__ == "__main__":
    prepare_meld_csvs()