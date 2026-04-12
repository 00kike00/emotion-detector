import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from transformers import RobertaTokenizer
import sys

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path: sys.path.append(str(root))

from src.architectures.text_net import RobertaBiLSTM
from src.config import PROCESSED_DIR, MELD_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_text_features(split_name):
    print(f"\n>>> Extracting Text Logits for: {split_name.upper()}")
    
    # 1. Load Tokenizer and Model
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaBiLSTM(num_classes=7).to(DEVICE)
    checkpoint = torch.load("models/weights/text_expert_best.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # 2. Load Text Data
    csv_path = PROCESSED_DIR / "train_split.csv" if split_name != "dev" else MELD_DIR / "dev/dev_sent_emo.csv"
    if split_name == "test": csv_path = PROCESSED_DIR / "test_split.csv"
    
    df = pd.read_csv(csv_path, encoding='latin-1')
    text_features = []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Text {split_name}"):
            text = str(row['Utterance']).replace('\u2019', "'")
            
            # Tokenizationn
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(DEVICE)
            
            # Inference: Obtain logits [1, 7]
            logits = model(inputs['input_ids'], inputs['attention_mask'])
            
            text_features.append({
                'id': f"{row['Dialogue_ID']}_{row['Utterance_ID']}",
                'text_logits': logits.cpu().squeeze(0), # Guardamos vector [7]
                'label': row['Emotion']
            })

    output_path = PROCESSED_DIR / f"meld_{split_name}_text_logits.pt"
    torch.save(text_features, output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    for s in ["train", "dev", "test"]:
        extract_text_features(s)