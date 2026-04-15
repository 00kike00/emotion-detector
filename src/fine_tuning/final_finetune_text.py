import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from transformers import RobertaTokenizer

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.config import DEVICE, PROCESSED_DIR, MODELS_DIR, MELD_DIR
from src.architectures.text_net import RobertaBiLSTM

EMOTION_MAPPING = {
    'neutral': 0, 'happiness': 1, 'surprise': 2, 'sadness': 3,
    'anger': 4, 'disgust': 5, 'fear': 6
}

EMOTION_NAMES = ['neutral', 'happiness', 'surprise',
                 'sadness', 'anger', 'disgust', 'fear']

CSV_PATHS = {
    "train": PROCESSED_DIR / "meld_train_split.csv",
    "dev":   MELD_DIR / "dev/dev_sent_emo.csv",
    "test":  PROCESSED_DIR / "meld_test_split.csv",
}

# ── Dataset ───────────────────────────────────────────────────────────────────
class MELDTextDataset(Dataset):
    def __init__(self, split_name: str, tokenizer: RobertaTokenizer):
        csv_path = CSV_PATHS[split_name]
        df = pd.read_csv(csv_path, encoding='latin-1')

        if split_name == "dev":
            df["Emotion"] = df["Emotion"].replace({"joy": "happiness"})

        self.samples = []
        skipped = 0

        for _, row in df.iterrows():
            emotion = row['Emotion']
            if emotion not in EMOTION_MAPPING:
                skipped += 1
                continue

            text = str(row['Utterance']).replace('\u2019', "'")
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=64
            )

            self.samples.append({
                'input_ids':      inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'label':          EMOTION_MAPPING[emotion]
            })

        print(f"Split {split_name}: {len(self.samples)} utterances loaded "
              f"| Skipped: {skipped}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s['input_ids'],
            s['attention_mask'],
            torch.tensor(s['label'], dtype=torch.long)
        )

    def get_labels(self):
        """Returns all labels — used for WeightedRandomSampler."""
        return [s['label'] for s in self.samples]


# ── Fine-Tuning Function ──────────────────────────────────────────────────────
def run_text_finetuning():
    # 1. TOKENIZER
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # 2. LOAD DATA
    train_ds = MELDTextDataset("train", tokenizer)
    dev_ds   = MELDTextDataset("dev",   tokenizer)

    # 3. WEIGHTED SAMPLER
    train_labels   = train_ds.get_labels()
    class_counts   = np.bincount(train_labels)
    class_weights  = 1.0 / class_counts.astype(np.float32)
    sample_weights = torch.tensor(class_weights[train_labels], dtype=torch.float32)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=64,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 4. LOAD PRETRAINED MODEL
    checkpoint  = torch.load(MODELS_DIR / "final_text_expert_best.pth", map_location=DEVICE)
    best_params = checkpoint['config']

    model = RobertaBiLSTM(
        num_classes=7,
        hidden_dim=int(best_params['best_hidden_units']),
        dropout=best_params['best_dropout'],
        pooling_mode=best_params['best_pooling_mode']
    ).to(DEVICE)

    model.load_state_dict(checkpoint['state_dict'])

    # 5. FREEZE ROBERTA — only finetune BiLSTM head
    for name, param in model.named_parameters():
        if 'roberta' in name:
            param.requires_grad = False

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Trainable layers: {trainable}")

    # 6. PLAIN LOSS + OPTIMIZER — sampler handles imbalance
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=2e-5
    )

    best_f1 = 0.0

    # 7. TRAINING LOOP
    for epoch in range(5):
        model.train()
        train_loss = 0

        for input_ids, attention_mask, labels in tqdm(
                train_loader, desc=f"Epoch {epoch+1}"):
            input_ids      = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels         = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)  # [B, 7]
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss   = 0
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for input_ids, attention_mask, labels in dev_loader:
                input_ids      = input_ids.to(DEVICE)
                attention_mask = attention_mask.to(DEVICE)
                labels         = labels.to(DEVICE)

                outputs   = model(input_ids, attention_mask)
                val_loss += nn.CrossEntropyLoss()(outputs, labels).item()

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(dev_loader)
        val_f1    = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train:.4f} | "
              f"Val Loss: {avg_val:.4f} | Val Macro F1: {val_f1:.4f}")

        # Save based on macro F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'config':     best_params  # preserve for extraction script
                },
                MODELS_DIR / "final_text_expert_best_ft.pth"
            )
            print(f">>> Saved: final_text_expert_best_ft.pth (Macro F1: {val_f1:.4f})")

        # See Finetune Progress each Epoch
        print(classification_report(
            all_labels, all_preds,
            target_names=EMOTION_NAMES,
            zero_division=0
        ))


if __name__ == "__main__":
    run_text_finetuning()