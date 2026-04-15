import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from transformers import RobertaTokenizer
import sys

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path: sys.path.append(str(root))

from src.architectures.text_net import RobertaBiLSTM
from src.config import PROCESSED_DIR, MELD_DIR, MODELS_DIR, DEVICE

EMOTION_MAPPING = {
    'neutral': 0, 'happiness': 1, 'surprise': 2, 'sadness': 3,
    'anger': 4, 'disgust': 5, 'fear': 6
    }

csv_paths = {
    "train": PROCESSED_DIR / "meld_train_split.csv",
    "dev":   MELD_DIR / "dev/dev_sent_emo.csv",
    "test":  PROCESSED_DIR / "meld_test_split.csv",
}


def extract_text_features(split_name):
    print(f"\n>>> Extracting Text Logits for: {split_name.upper()}")
    
    # 1. Load Tokenizer and Model
    checkpoint = torch.load(MODELS_DIR / "final_text_expert_best_ft.pth", map_location=DEVICE)
    best_params = checkpoint['config']
    
    # Initialize Model 
    model = RobertaBiLSTM(
        num_classes=7,
        hidden_dim=int(best_params['best_hidden_units']),
        dropout=best_params['best_dropout'],
        pooling_mode=best_params['best_pooling_mode']
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # 2. Load Text Data
    csv_path = csv_paths[split_name]
    
    df = pd.read_csv(csv_path, encoding='latin-1')
    if split_name == "dev":
        df["Emotion"] = df["Emotion"].replace({"joy": "happiness"}) 
    text_features = []

    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Text {split_name}"):
            text = str(row['Utterance']).replace('\u2019', "'")
            
            # Tokenization
            inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=32).to(DEVICE)
            
            # Inference: Obtain logits [1, 7]
            logits = model(inputs['input_ids'], inputs['attention_mask'])
            
            text_features.append({
                'id': f"{row['Dialogue_ID']}_{row['Utterance_ID']}",
                'text_logits': logits.cpu().squeeze(0), # Save vector [7]
                'label': EMOTION_MAPPING[row['Emotion']]
            })

    output_path = PROCESSED_DIR / f"final_meld_{split_name}_text_logits_ft.pt"
    torch.save(text_features, output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    for s in ["train", "dev", "test"]:
        extract_text_features(s)