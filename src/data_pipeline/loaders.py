import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
import torchaudio
import torchaudio.transforms as AT
from transformers import RobertaTokenizer
from PIL import Image
from src.config import FER_DIR, GOEMOTIONS_DIR, RAVDESS_DIR, BATCH_SIZE, PROCESSED_DIR
import json
from pathlib import Path
import soundfile as sf
import random

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
        # 1. Horizontal Flip (faces can be mirrored in videos)
        transforms.RandomHorizontalFlip(p=0.5),
        # 2. Small Rotation (for tilted faces in MELD)
        transforms.RandomRotation(degrees=10),
        # 3. Variation of Lighting and Contrast (critical for video)
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        # 4. Small Translations or Zooms (resiliency to framing)
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        # Normalization: FER2013 pixel values are [0, 255], we normalize to [-1, 1]
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

    # Compute per-sample weights from class frequencies
    labels = train_ds.data['label'].values
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = torch.tensor(class_weights[labels], dtype=torch.float32)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
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


def random_word_swap(text, p=0.1):
    words = text.split()
    if len(words) < 2:
        return text
    
    # Decidimos cuántas veces swappear (mínimo 1 si se activa)
    n = max(1, int(len(words) * p))
    
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(words)), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
        
    return " ".join(words)

class TextEmotionDataset(Dataset):
    EMOTION_TO_IDX = {
        'neutral': 0, 'happiness': 1, 'surprise': 2, 'sadness': 3, 
        'anger': 4, 'disgust': 5, 'fear': 6
    }

    def __init__(self, tsv_path, tokenizer, mapping_path, emotions_path, max_len=64, augment=False):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment

        # 1. Load the 28 emotion names from emotions.txt
        with open(emotions_path, 'r') as f:
            idx_to_goemotions = [line.strip() for line in f if line.strip()]

        # 2. Load and invert mapping (28 emotions -> 7 Ekman)
        with open(mapping_path, 'r') as f:
            mapping_json = json.load(f)

        # Invert: {'anger': ['anger', 'annoyance']} -> {'anger': 'anger', 'annoyance': 'anger'}
        go_to_7 = {}
        for seven_class, go_list in mapping_json.items():
            for go_name in go_list:
                go_to_7[go_name] = seven_class
        
        # Ensure 'neutral' is mapped 
        if 'neutral' not in go_to_7:
            go_to_7['neutral'] = 'neutral'

        # 3. Load Raw Data
        df_raw = pd.read_csv(tsv_path, sep='\t', header=None, names=['text', 'labels', 'id'])

        # 4. Map and Filter
        valid_rows = []
        for _, row in df_raw.iterrows():
            # Get first label index and convert to name
            first_idx = int(str(row['labels']).split(',')[0])
            go_name = idx_to_goemotions[first_idx]
            
            # Convert name to 7-class name
            mapped_name = go_to_7.get(go_name)
            
            # Convert 7-class name to numeric ID (0-6)
            if mapped_name in self.EMOTION_TO_IDX:
                valid_rows.append({
                    'text': row['text'],
                    'label': self.EMOTION_TO_IDX[mapped_name]
                })

        self.df = pd.DataFrame(valid_rows)
        print(f"[Dataset] {Path(tsv_path).name} | Samples: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        text = str(row.text)
        label = row.label

        # Random Word Swap Augmentation (only for training)
        prob = 0.25 if label in [5, 6] else 0.10  # More augmentation for disgust/fear
        if self.augment and random.random() < prob:
            text = random_word_swap(text, p=0.1)

        encoding = self.tokenizer(
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
    
    # Define paths
    mapping = GOEMOTIONS_DIR / "ekman_mapping.json"
    emotions = GOEMOTIONS_DIR / "emotions.txt"

    train_ds = TextEmotionDataset(GOEMOTIONS_DIR / "train.tsv", tokenizer, mapping, emotions, augment=True)
    val_ds   = TextEmotionDataset(GOEMOTIONS_DIR / "dev.tsv",   tokenizer, mapping, emotions, augment=False)
    test_ds  = TextEmotionDataset(GOEMOTIONS_DIR / "test.tsv",  tokenizer, mapping, emotions, augment=False)

    # Compute per-sample weights from class frequencies
    labels = train_ds.df['label'].values
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = torch.tensor(class_weights[labels], dtype=torch.float32)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    # BATCH_SIZE//2 is used here to reduce GPU memory usage, since text models can be large. Adjust as needed.
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE//2, sampler=sampler, 
        num_workers=8, pin_memory=True, 
        persistent_workers=True, prefetch_factor=4
    )
    val_loader   = DataLoader(
        val_ds, batch_size=BATCH_SIZE//2, shuffle=True, 
        num_workers=4, pin_memory=True, 
        persistent_workers=True, prefetch_factor=4
    )
    test_loader  = DataLoader(
        test_ds, batch_size=BATCH_SIZE//2, shuffle=True, 
        num_workers=4, pin_memory=True, 
        persistent_workers=True, prefetch_factor=4
    )

    return train_loader, val_loader, test_loader



SAMPLE_RATE    = 22050
N_MELS         = 128
N_FFT          = 1024
HOP_LENGTH     = 512
MAX_FRAMES     = 172  # fixed width — pad or crop spectrogram to this


class RAVDESSDataset(Dataset):
    RAVDESS_EMOTIONS = {
    '01': 'neutral',
    '03': 'happiness',
    '04': 'sadness',
    '05': 'anger',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprise',
    }  # '02' calm is dropped

    TARGET_EMOTIONS = ['neutral', 'happiness', 'surprise', 'sadness',
                    'anger', 'disgust', 'fear']
    EMOTION_TO_IDX = {e: i for i, e in enumerate(TARGET_EMOTIONS)}

    def __init__(self, file_list, augment=False):
        self.file_list = file_list
        self.augment   = augment

        self.mel = AT.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )
        self.to_db = AT.AmplitudeToDB()

        # SpecAugment — applied on spectrogram during augmentation
        self.time_mask = AT.TimeMasking(time_mask_param=20)
        self.freq_mask = AT.FrequencyMasking(freq_mask_param=8)

    def __len__(self):
        return len(self.file_list)

    def _parse_label(self, path):
        parts = path.stem.split('-')
        emotion_code = parts[2]
        return self.RAVDESS_EMOTIONS.get(emotion_code, None)

    def _load_and_resample(self, path):
        waveform, sr = sf.read(str(path), dtype='float32')
    
        # soundfile returns [samples] or [samples, channels] — convert to [channels, samples]
        if waveform.ndim == 1:
            waveform = waveform[None, :]          # [1, samples]
        else:
            waveform = waveform.T                 # [channels, samples]
        
        waveform = torch.tensor(waveform)
        
        # Mix to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
        
        return waveform

    def _add_noise(self, waveform, snr_db=20):
        noise = torch.randn_like(waveform)
        signal_power = waveform.norm(p=2)
        noise_power  = noise.norm(p=2)
        scale = (signal_power / noise_power) * (10 ** (-snr_db / 20))
        return waveform + scale * noise

    def _to_spectrogram(self, waveform):
        spec = self.mel(waveform)        # [1, N_MELS, T]
        spec = self.to_db(spec)

        # Normalize to [0, 1]
        spec = spec = (spec + 40) / 40  # RAVDESS has a min around -40dB, so this roughly scales to [0, 1]

        # Pad or crop to fixed width
        T = spec.shape[2]
        if T < MAX_FRAMES:
            spec = torch.nn.functional.pad(spec, (0, MAX_FRAMES - T))
        else:
            spec = spec[:, :, :MAX_FRAMES]

        return spec  # [1, N_MELS, MAX_FRAMES]

    def __getitem__(self, idx):
        path  = self.file_list[idx]
        label = self._parse_label(path)

        waveform = self._load_and_resample(path)

        if self.augment:
            if torch.rand(1) < 0.5:
                waveform = self._add_noise(waveform, snr_db=20)

        spec = self._to_spectrogram(waveform)

        if self.augment:
            if torch.rand(1) < 0.5:
                spec = self.time_mask(spec)
            if torch.rand(1) < 0.5:
                spec = self.freq_mask(spec)

        return spec, torch.tensor(self.EMOTION_TO_IDX[label], dtype=torch.long)


def get_ravdess_files():
    """Collect audio-only speech files, drop calm (02)."""
    RAVDESS_EMOTIONS = {
    '01': 'neutral',
    '03': 'happiness',
    '04': 'sadness',
    '05': 'anger',
    '06': 'fear',
    '07': 'disgust',
    '08': 'surprise',
    }  # '02' calm is dropped
    all_files = []
    for actor_dir in sorted(RAVDESS_DIR.iterdir()):
        if not actor_dir.is_dir():
            continue
        for wav in actor_dir.glob('*.wav'):
            parts = wav.stem.split('-')
            emotion  = parts[2]
            if emotion not in RAVDESS_EMOTIONS:  # drops calm
                continue
            all_files.append(wav)
    return all_files


def get_ravdess_loaders(val_actors=(23, 24), test_actors=(21, 22)):
    """
    Speaker-independent split: hold out 2 actors for val, 2 for test.
    """
    all_files = get_ravdess_files()

    train_files, val_files, test_files = [], [], []
    for f in all_files:
        actor_id = int(f.stem.split('-')[6])
        if actor_id in test_actors:
            test_files.append(f)
        elif actor_id in val_actors:
            val_files.append(f)
        else:
            train_files.append(f)

    print(f"[RAVDESS] Train={len(train_files)} | Val={len(val_files)} | Test={len(test_files)}")

    train_ds = RAVDESSDataset(train_files, augment=True)
    val_ds   = RAVDESSDataset(val_files,   augment=False)
    test_ds  = RAVDESSDataset(test_files,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE//8, shuffle=True,
                              num_workers=8, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE//8, shuffle=False,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE//8, shuffle=False,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)

    return train_loader, val_loader, test_loader


class MELDLogitsDataset(Dataset):
    def __init__(self, split_name: str):
        # Load both modality files
        vision_data = torch.load(PROCESSED_DIR / f"meld_{split_name}_vision_logits_ft.pt")
        text_data   = torch.load(PROCESSED_DIR / f"meld_{split_name}_text_logits.pt")

        # Build id -> text_logits lookup for fast joining
        text_lookup = {s['id']: s['text_logits'] for s in text_data}

        self.samples = []
        skipped = 0

        for v_sample in vision_data:
            uid = v_sample['id']
            if uid not in text_lookup:
                skipped += 1
                continue

            self.samples.append({
                'vision_seq':  v_sample['vision_logits'],  # [T, 7]
                'text_logits': text_lookup[uid],            # [7]
                'label':       v_sample['label'],           # int
            })

        print(f"[MELDLogitsDataset] split={split_name} | "
              f"Loaded={len(self.samples)} | Skipped={skipped} (id mismatch)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s['vision_seq'],                               # [T, 7]  — variable T
            s['text_logits'],                              # [7]
            torch.tensor(s['label'], dtype=torch.long)    # scalar
        )


def collate_fn(batch):
    """
    Pads variable-length vision sequences within a batch.
    Returns:
        vision_seqs  : [B, T_max, 7]  — padded
        lengths      : [B]            — original T per sample (for pack_padded if needed)
        text_logits  : [B, 7]
        labels       : [B]
    """
    vision_seqs, text_logits, labels = zip(*batch)

    lengths     = torch.tensor([s.shape[0] for s in vision_seqs], dtype=torch.long)
    vision_pad  = pad_sequence(vision_seqs, batch_first=True, padding_value=0.0)  # [B, T_max, 7]
    text_stack  = torch.stack(text_logits)   # [B, 7]
    label_stack = torch.stack(labels)        # [B]

    return vision_pad, lengths, text_stack, label_stack


def get_meld_loaders():
    train_ds = MELDLogitsDataset("train")
    dev_ds   = MELDLogitsDataset("dev")
    test_ds  = MELDLogitsDataset("test")

    # Weighted sampler for train to handle class imbalance
    labels       = [s['label'] for s in train_ds.samples]
    class_counts = np.bincount(labels)
    class_weights  = 1.0 / class_counts
    sample_weights = torch.tensor(class_weights[labels], dtype=torch.float32)

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        collate_fn=collate_fn, num_workers=8,
        pin_memory=True, persistent_workers=True
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=4,
        pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=4,
        pin_memory=True, persistent_workers=True
    )

    return train_loader, dev_loader, test_loader