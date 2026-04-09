import torch
import cv2
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from transformers import RobertaTokenizer
from torchvision import transforms
import sys

# Path Setup
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

# Project Imports
from src.config import DEVICE, MODELS_DIR, MELD_DIR, PROCESSED_DIR, CHECKPOINTS_DIR
from src.architectures.text_net import RobertaBiLSTM
from src.architectures.vision_net import VisionNet

def get_largest_face(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    # Pick the face with the largest area (w * h) assuming it's the main speaker in the most cases
    best_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = best_face
    return gray[y:y+h, x:x+w]

def run_extraction(split_name):
    print(f"\n>>> Starting Extraction for: {split_name.upper()}")
    
    # 1. SETUP PATHS
    split_dir = MELD_DIR / split_name
    csv_path = split_dir / f"{split_name}_sent_emo.csv"
    video_dir = split_dir / f"{split_name}_splits_complete"
    output_path = PROCESSED_DIR / f"meld_{split_name}_features.pt"

    # 2. LOAD EXPERTS
    # --- Text Expert ---
    text_ckpt = torch.load(MODELS_DIR / "text_expert_best.pth", map_location=DEVICE)
    text_model = RobertaBiLSTM(num_classes=7, **text_ckpt['config']).to(DEVICE)
    text_model.load_state_dict(text_ckpt['state_dict'])
    text_model.eval()
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    # --- Vision Expert ---
    import json
    with open(CHECKPOINTS_DIR/ "vision_apso_results.json", "r") as f:
        v_params = json.load(f)
    
    vision_model = VisionNet(
        dropout_rate=v_params['best_dropout'], 
        hidden_units=v_params['best_hidden_units']
    ).to(DEVICE)
    vision_model.load_state_dict(torch.load(MODELS_DIR / "vision_expert_best.pth", map_location=DEVICE))
    vision_model.eval()

    # 3. PREPROCESSING UTILS
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    v_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mapping = {
        'neutral': 0, 'joy': 1, 'surprise': 2, 'sadness': 3, 
        'anger': 4, 'disgust': 5, 'fear': 6
    }

    # 4. DATA PROCESSING
    df = pd.read_csv(csv_path, encoding='latin-1')
    df['Utterance'] = df['Utterance'].str.replace('’', "'", regex=False)
    
    extracted_data = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        # --- A. Text Features ---
        t_inputs = tokenizer(row['Utterance'], return_tensors="pt", padding='max_length', 
                             truncation=True, max_length=64).to(DEVICE)
        with torch.no_grad():
            t_logits = text_model(t_inputs['input_ids'], t_inputs['attention_mask'])

        # --- B. Vision Features (Mean Logits) ---
        v_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
        v_path = video_dir / v_filename
        
        v_all_logits = []
        if v_path.exists():
            cap = cv2.VideoCapture(str(v_path))
            # Sample every 5th frame for speed/coverage
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if frame_idx % 5 == 0:
                    face = get_largest_face(frame, face_cascade)
                    if face is not None:
                        face_tensor = v_transform(face).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            logits = vision_model(face_tensor)
                            v_all_logits.append(logits)
                frame_idx += 1
            cap.release()

        # Average the vision logits across the video
        if v_all_logits:
            v_mean_logits = torch.mean(torch.stack(v_all_logits), dim=0)
        else:
            v_mean_logits = torch.zeros(1, 7).to(DEVICE) # Zero vector if no face found

        # --- C. Save to List ---
        extracted_data.append({
            'text_feat': t_logits.cpu().squeeze(0),      # Size [7]
            'vision_feat': v_mean_logits.cpu().squeeze(0), # Size [7]
            'label': mapping[row['Emotion']]
        })

    torch.save(extracted_data, output_path)
    print(f"Finished! Saved to {output_path}")

if __name__ == "__main__":
    # Run for both original sets
    run_extraction("train")
    run_extraction("dev")