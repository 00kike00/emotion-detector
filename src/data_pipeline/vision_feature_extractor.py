import torch
from tqdm import tqdm
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path: 
    sys.path.append(str(root))

from src.architectures.vision_net import VisionNet
from src.config import PROCESSED_DIR, MODELS_DIR, CHECKPOINTS_DIR, DEVICE

def extract_vision_features(split_name):
    print(f"\n>>> Extracting Vision Logits for: {split_name.upper()}")
    
    # LOAD HYPERPARAMETERS & MODEL
    with open(CHECKPOINTS_DIR / "vision_apso_acc_results.json", "r") as f:
        best_params = json.load(f)

    model = VisionNet(
        dropout_rate=best_params['best_dropout'], 
        hidden_units=best_params['best_hidden_units']
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODELS_DIR / "vision_expert_best_acc.pth", map_location=DEVICE))
    model.eval()

    # 2. Load Face Data
    data_path = PROCESSED_DIR / f"meld_{split_name}_faces.pt"
    samples = torch.load(data_path)
    
    vision_features = []

    with torch.no_grad():
        for sample in tqdm(samples, desc=f"Vision {split_name}"):
            # frames shape: [T, 1, 48, 48]
            frames = sample['frames'].to(DEVICE)
            
            # Inference: Obtain logits [T, 7]
            logits = model(frames) 
            
            vision_features.append({
                'id': f"{sample['dia_id']}_{sample['utt_id']}",
                'vision_logits': logits.cpu(),
                'label': sample['label']
            })

    output_path = PROCESSED_DIR / f"meld_{split_name}_vision_logits.pt"
    torch.save(vision_features, output_path)
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    for s in ["train", "dev", "test"]:
        extract_vision_features(s)