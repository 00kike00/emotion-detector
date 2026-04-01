import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import RobertaTokenizer
from PIL import Image
from src.config import FER_DIR, GOEMOTIONS_DIR, BATCH_SIZE

class FERPlusWinnerDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, usage='Training'):
        df = pd.read_csv(csv_file)
        df = df[df['Usage'] == usage].copy()

        self.target_emotions = ['neutral', 'happiness', 'surprise', 'sadness', 
                                'anger', 'disgust', 'fear']
        
        # All columns including the ones we want to discard
        all_vote_cols = self.target_emotions + ['contempt', 'unknown', 'NF']
        
        # Find the winner (most voted) across ALL columns per row
        winner = df[all_vote_cols].idxmax(axis=1)
        
        # Only keep rows where winner is one of our 7 emotions
        mask = winner.isin(self.target_emotions)
        self.data = df[mask].copy()
        self.data['label'] = winner[mask].map(
            {e: i for i, e in enumerate(self.target_emotions)}
        )
        
        self.img_dir = img_dir
        self.transform = transform
        
        print(f"[Dataset] Usage={usage} | Kept={len(self.data)} | "
              f"Discarded={len(df) - len(self.data)} (contempt/unknown/NF winners)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['Image name']
        label = self.data.iloc[idx]['label']
        
        img_path = self.img_dir / img_name
        image = Image.open(img_path).convert('L')  # Grayscale

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_fer_loaders():
    train_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    csv_path = FER_DIR / "fer2013new.csv"
    
    train_ds = FERPlusWinnerDataset(csv_path, FER_DIR / "FER2013Train", train_transform, usage='Training')
    valid_ds = FERPlusWinnerDataset(csv_path, FER_DIR / "FER2013Valid", val_transform, usage='PublicTest')
    test_ds  = FERPlusWinnerDataset(csv_path, FER_DIR / "FER2013Test",  val_transform, usage='PrivateTest')

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=4
    )

    return train_loader, valid_loader, test_loader


class TextEmotionDataset(Dataset):
    def __init__(self, tsv_path, tokenizer, max_len=64):
        # GoEmotions TSV structure: [text, label_ids, id]
        self.df = pd.read_csv(tsv_path, sep='\t', header=None, names=['text', 'labels', 'id'])
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Simple Mapping logic (You can refine this based on your ekman_mapping.json)
        # For now, we take the first label if multiple exist
        self.df['primary_label'] = self.df['labels'].apply(lambda x: int(str(x).split(',')[0]))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        text = str(self.df.iloc[item].text)
        label = self.df.iloc[item].primary_label

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def get_text_loaders():
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    train_ds = TextEmotionDataset(GOEMOTIONS_DIR / "train.tsv", tokenizer)
    test_ds = TextEmotionDataset(GOEMOTIONS_DIR / "test.tsv", tokenizer)
    val_ds = TextEmotionDataset(GOEMOTIONS_DIR / "validation.tsv", tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    return train_loader, val_loader, test_loader