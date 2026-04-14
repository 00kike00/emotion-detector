import torch
import cv2
import json
import sys
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image

# 1. PATH SETUP
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.config import DEVICE, CHECKPOINTS_DIR, MODELS_DIR
from src.architectures.vision_net import VisionNet

def run_live_vision():
    # 2. LOAD HYPERPARAMETERS & MODEL
    with open(CHECKPOINTS_DIR / "final_vision_apso_results.json", "r") as f:
        best_params = json.load(f)

    model = VisionNet(
        dropout_rate=best_params['best_dropout'], 
        hidden_units=best_params['best_hidden_units']
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODELS_DIR / "final_vision_expert_best.pth", map_location=DEVICE))
    model.eval()

    # 3. SETUP FACE DETECTION & TRANSFORMS
    # OpenCV's built-in face finder
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    emotions = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear']
    
    # 4. START WEBCAM
    cap = cv2.VideoCapture(0) # 0 is usually the default webcam
    print("Press 'q' to quit the live feed.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Convert to grayscale for the face detector
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Crop and Preprocess the face
            face_roi = gray_frame[y:y+h, x:x+w]
            pil_img = Image.fromarray(face_roi)
            input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

            # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                prob = torch.nn.functional.softmax(outputs, dim=1)
                top_p, top_class = torch.max(prob, 1)
            
            emotion = emotions[top_class.item()]
            conf = top_p.item() * 100

            # Display Text
            label = f"{emotion} ({conf:.1f}%)"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        cv2.imshow('Emotion Detector - Vision Expert', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_vision()