import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from src.config import FER_DIR, BATCH_SIZE

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from src.config import FER_DIR, BATCH_SIZE

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