import torch
import torch.nn.functional as F
import json
import sys
from pathlib import Path
from transformers import RobertaTokenizer

# 1. PATH SETUP
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.config import DEVICE, MODELS_DIR
from src.architectures.text_net import RobertaBiLSTM

# Silence RoBERTa loading warnings
from transformers import logging
logging.set_verbosity_error()

def run_text_inference():
    # 2. LOAD SAVED MODEL & CONFIG
    # Note: We load the .pth we saved in train_final, which contains the config
    checkpoint_path = MODELS_DIR / "final_text_expert_best_ft.pth"
    
    if not checkpoint_path.exists():
        print(f"Error: {checkpoint_path} not found. Please train the model first!")
        return

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
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

    # 3. SETUP TOKENIZER
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    emotions = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear']
    
    print("\n--- Text Emotion Expert Loaded ---")
    print("Type your message and press Enter to see the predicted emotion.")
    print("Type 'q' or 'exit' to quit.")
    print("-" * 35)

    # 4. INFERENCE LOOP
    while True:
        user_input = input("\nUser: ")
        
        if user_input.lower() in ['q', 'exit', 'quit']:
            break
            
        if not user_input.strip():
            continue

        # Preprocess / Tokenize
        # We use max_length=64 or 32 based on what you used in training
        inputs = tokenizer(
            user_input,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=64 
        ).to(DEVICE)

        # Inference
        with torch.no_grad():
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            probs = F.softmax(outputs, dim=1)
            top_p, top_class = torch.max(probs, 1)

        emotion = emotions[top_class.item()]
        confidence = top_p.item() * 100

        # 5. DISPLAY RESULTS
        color_code = "\033[92m" # Green for display
        reset_code = "\033[0m"
        
        print(f"Prediction: {color_code}{emotion}{reset_code} ({confidence:.1f}%)")

if __name__ == "__main__":
    run_text_inference()