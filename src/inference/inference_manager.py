import torch
import cv2
import json
import sys
import numpy as np
from pathlib import Path
from transformers import RobertaTokenizer

# 1. PATH SETUP
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.config import DEVICE, CHECKPOINTS_DIR, MODELS_DIR
from src.architectures.vision_net import VisionNet
from src.architectures.text_net import RobertaBiLSTM
from src.architectures.manager_net import ManagerNet

# ── Constants ─────────────────────────────────────────────────────────────────
EMOTIONS       = ['Neutral', 'Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear']
DNN_PROTOTXT   = Path("models/face_detection/deploy.prototxt")
DNN_WEIGHTS    = Path("models/face_detection/res10_300x300_ssd_iter_140000.caffemodel")
CONFIDENCE_THR = 0.5
FRAME_STEP     = 5


def load_models():
    # Vision expert
    with open(CHECKPOINTS_DIR / "final_vision_apso_results.json", "r") as f:
        vp = json.load(f)
    vision = VisionNet(
        dropout_rate=vp['best_dropout'],
        hidden_units=vp['best_hidden_units']
    ).to(DEVICE)
    ckpt_v = torch.load(
        MODELS_DIR / "final_vision_expert_best_ft.pth", map_location=DEVICE
    )
    vision.load_state_dict(ckpt_v['state_dict'])
    vision.eval()

    # Text expert
    ckpt_t      = torch.load(
        MODELS_DIR / "final_text_expert_best.pth", map_location=DEVICE
    )
    best_params = ckpt_t['config']
    text = RobertaBiLSTM(
        num_classes=7,
        hidden_dim=int(best_params['best_hidden_units']),
        dropout=best_params['best_dropout'],
        pooling_mode=best_params['best_pooling_mode']
    ).to(DEVICE)
    text.load_state_dict(ckpt_t['state_dict'])
    text.eval()

    # Manager
    ckpt_m = torch.load(
        MODELS_DIR / "final_manager_best.pth", map_location=DEVICE
    )
    mp = ckpt_m['config']
    manager = ManagerNet(
        num_classes=7,
        vision_input_dim=7,
        text_input_dim=7,
        hidden_dim=mp['best_hidden_units'],
        num_layers=mp['best_num_layers'],
        dropout_rate=mp['best_dropout'],
    ).to(DEVICE)
    manager.load_state_dict(ckpt_m['state_dict'])
    manager.eval()

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    dnn       = cv2.dnn.readNetFromCaffe(str(DNN_PROTOTXT), str(DNN_WEIGHTS))

    return vision, text, manager, tokenizer, dnn


def detect_face(frame_bgr, dnn):
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame_bgr, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    dnn.setInput(blob)
    detections = dnn.forward()

    best_conf, best_crop, best_box = 0.0, None, None
    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < CONFIDENCE_THR:
            continue
        x1 = max(0, int(detections[0, 0, i, 3] * w))
        y1 = max(0, int(detections[0, 0, i, 4] * h))
        x2 = min(w, int(detections[0, 0, i, 5] * w))
        y2 = min(h, int(detections[0, 0, i, 6] * h))
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        if conf > best_conf:
            best_conf = conf
            best_crop = crop
            best_box  = (x1, y1, x2, y2)

    return best_crop, best_box


def preprocess_face(crop_bgr):
    gray    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    tensor  = torch.tensor(
        resized, dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0) / 255.0
    return ((tensor - 0.5) / 0.5).to(DEVICE)


def infer_text(text_input, tokenizer, text_model):
    inputs = tokenizer(
        text_input,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=64
    )
    with torch.no_grad():
        return text_model(
            inputs['input_ids'].to(DEVICE),
            inputs['attention_mask'].to(DEVICE)
        )


def draw_text_block(frame, lines, x, y, font_scale=0.55,
                    color=(200, 210, 245), thickness=1,
                    bg_color=(20, 20, 40), padding=6):
    """
    Draws multiple lines of text with a semi-transparent background block.
    """
    font      = cv2.FONT_HERSHEY_SIMPLEX
    line_h    = int(font_scale * 30) + 4
    max_w     = max(cv2.getTextSize(l, font, font_scale, thickness)[0][0]
                    for l in lines) + padding * 2
    block_h   = line_h * len(lines) + padding * 2

    # Dark background rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + max_w, y + block_h), bg_color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    for i, line in enumerate(lines):
        ty = y + padding + (i + 1) * line_h - 4
        cv2.putText(frame, line, (x + padding, ty),
                    font, font_scale, color, thickness, cv2.LINE_AA)


def run_live_manager():
    print("Loading models...")
    vision_model, text_model, manager, tokenizer, dnn = load_models()
    print("Models loaded.")
    print("Controls: type your message directly | ENTER to send | BACKSPACE to delete | Q to quit")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # ── State ─────────────────────────────────────────────────────────────────
    gru_hidden           = None
    vision_logit_buffer  = []
    frame_counter        = 0

    # Display state
    vision_emotion   = "—"
    manager_emotion  = "—"
    manager_conf     = 0.0
    text_emotion     = "—"
    anchored_emotion = "—"
    anchored_conf    = 0.0

    # Text input state — typed directly in the OpenCV window
    current_input    = ""       # what user is currently typing
    last_sent        = ""       # last message that was sent
    last_result_lines = []      # result lines shown after sending

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_counter += 1

        # ── Vision inference ──────────────────────────────────────────────────
        if frame_counter % FRAME_STEP == 0:
            crop, box = detect_face(frame, dnn)

            if crop is not None and box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (92, 104, 152), 2)

                face_tensor  = preprocess_face(crop)

                with torch.no_grad():
                    frame_logits = vision_model(face_tensor)
                    v_probs      = torch.softmax(
                        frame_logits.squeeze(), dim=-1
                    ).cpu().numpy()
                    vision_emotion = EMOTIONS[int(np.argmax(v_probs))]

                    vision_logit_buffer.append(
                        frame_logits.squeeze(0).cpu()
                    )

                    logit_input = frame_logits.squeeze(0).unsqueeze(0)
                    manager_logits, gru_hidden = manager.forward_streaming(
                        logit_input, gru_hidden
                    )
                    m_probs         = torch.softmax(
                        manager_logits.squeeze(), dim=-1
                    ).cpu().numpy()
                    manager_emotion = EMOTIONS[int(np.argmax(m_probs))]
                    manager_conf    = float(np.max(m_probs))

        h_frame, w_frame = frame.shape[:2]

        # ── Top-left: live emotion status ─────────────────────────────────────
        status_lines = [
            f"Vision  : {vision_emotion}",
            f"Manager : {manager_emotion} ({manager_conf:.0%})  "
            f"[{len(vision_logit_buffer)} frames]",
            f"Text    : {text_emotion}",
            f"Anchored: {anchored_emotion} ({anchored_conf:.0%})",
        ]
        draw_text_block(frame, status_lines, x=10, y=10)

        # ── Bottom: text input bar ────────────────────────────────────────────
        input_bar_y = h_frame - 60
        cursor      = "_" if (frame_counter // 15) % 2 == 0 else " "
        input_lines = [
            f"Type: {current_input}{cursor}",
            f"Last: {last_sent[:60]}{'...' if len(last_sent) > 60 else ''}",
        ]
        draw_text_block(
            frame, input_lines,
            x=10, y=input_bar_y,
            font_scale=0.6,
            color=(220, 230, 255),
            bg_color=(10, 10, 30)
        )

        # ── Bottom-right: last result ─────────────────────────────────────────
        if last_result_lines:
            draw_text_block(
                frame, last_result_lines,
                x=w_frame - 320, y=h_frame - 120,
                font_scale=0.5,
                color=(180, 255, 180),
                bg_color=(10, 30, 10)
            )

        cv2.imshow("Manager Emotion Inference  |  Q=quit", frame)

        # ── Keyboard input ────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:   # Q or ESC — quit
            break

        elif key == 13:                    # ENTER — send message
            if current_input.strip() and gru_hidden is not None:
                user_text   = current_input.strip()
                last_sent   = user_text
                current_input = ""

                # Text expert
                text_logits  = infer_text(user_text, tokenizer, text_model)
                t_probs      = torch.softmax(
                    text_logits.squeeze(), dim=-1
                ).cpu().numpy()
                text_emotion = EMOTIONS[int(np.argmax(t_probs))]

                # Manager anchored mode
                last_frame = (
                    vision_logit_buffer[-1].unsqueeze(0).to(DEVICE)
                    if vision_logit_buffer
                    else torch.zeros(1, 7).to(DEVICE)
                )

                with torch.no_grad():
                    anchored_logits = manager.forward_with_text(
                        last_frame, gru_hidden, text_logits
                    )

                a_probs          = torch.softmax(
                    anchored_logits.squeeze(), dim=-1
                ).cpu().numpy()
                anchored_emotion = EMOTIONS[int(np.argmax(a_probs))]
                anchored_conf    = float(np.max(a_probs))

                # Update result overlay
                last_result_lines = [
                    f">> {user_text[:35]}{'...' if len(user_text) > 35 else ''}",
                    f"Vision  : {vision_emotion}",
                    f"Text    : {text_emotion}",
                    f"Manager : {anchored_emotion} ({anchored_conf:.0%})",
                ]

                # Reset GRU for next utterance
                gru_hidden          = None
                vision_logit_buffer = []

            elif current_input.strip() and gru_hidden is None:
                last_result_lines = ["No face detected yet — look at camera first"]
                current_input = ""

        elif key == 8:                     # BACKSPACE
            current_input = current_input[:-1]

        elif 32 <= key <= 126:             # printable ASCII
            current_input += chr(key)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_live_manager()