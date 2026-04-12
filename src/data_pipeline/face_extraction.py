import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms
import sys

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.config import MELD_DIR, PROCESSED_DIR

# ── Constants ────────────────────────────────────────────────────────────────
FRAME_STEP      = 5
DNN_PROTOTXT    = Path("models/face_detection/deploy.prototxt")
DNN_WEIGHTS     = Path("models/face_detection/res10_300x300_ssd_iter_140000.caffemodel")
CONFIDENCE_THR  = 0.5

# Matches FER training pipeline exactly: grayscale, 48x48, normalized to (-1,1)
face_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),                        # [0, 1]
    transforms.Normalize((0.5,), (0.5,))          # (-1, 1)
])

# Maps each split name to its CSV and video folder
SPLIT_CONFIG = {
    "train": {
        "csv":       PROCESSED_DIR / "meld_train_split.csv",
        "video_dir": MELD_DIR / "train/train_splits",
    },
    "dev": {
        "csv":       MELD_DIR / "dev/dev_sent_emo.csv",
        "video_dir": MELD_DIR / "dev/dev_splits_complete",
    },
    "test": {
        "csv":       PROCESSED_DIR / "meld_test_split.csv",
        "video_dir": MELD_DIR / "train/train_splits",  # same folder as train
    },
}

EMOTION_MAPPING = {
    'neutral': 0, 'happiness': 1, 'surprise': 2, 'sadness': 3,
    'anger': 4, 'disgust': 5, 'fear': 6
}


# ── DNN face detector ────────────────────────────────────────────────────────
def load_dnn_detector() -> cv2.dnn_Net:
    net = cv2.dnn.readNetFromCaffe(str(DNN_PROTOTXT), str(DNN_WEIGHTS))
    return net


def get_dominant_face_crop(
    frame_bgr: np.ndarray,
    net: cv2.dnn_Net,
) -> np.ndarray | None:
    """
    Run OpenCV DNN detector on a BGR frame.
    Returns the highest-confidence face crop (BGR) or None.
    """
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame_bgr, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()   # shape: [1, 1, N, 7]

    best_conf = 0.0
    best_crop = None

    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < CONFIDENCE_THR:
            continue

        x1 = max(0, int(detections[0, 0, i, 3] * w))
        y1 = max(0, int(detections[0, 0, i, 4] * h))
        x2 = min(w, int(detections[0, 0, i, 5] * w))
        y2 = min(h, int(detections[0, 0, i, 6] * h))

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        if confidence > best_conf:
            best_conf = confidence
            best_crop = crop

    return best_crop


# ── Per-video extraction ─────────────────────────────────────────────────────
def extract_frames_from_video(
    v_path: Path,
    net: cv2.dnn_Net,
) -> list[torch.Tensor]:
    """
    Sample every FRAME_STEP-th frame, detect face, apply transform.
    Returns list of [1, 48, 48] tensors. Empty list if no faces found.
    """
    cap = cv2.VideoCapture(str(v_path))
    tensors = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STEP == 0:
            crop = get_dominant_face_crop(frame_bgr, net)
            if crop is not None:
                tensor = face_transform(crop)   # [1, 48, 48]
                tensors.append(tensor)

        frame_idx += 1

    cap.release()
    return tensors


# ── Main extraction loop ─────────────────────────────────────────────────────
def run_extraction(split_name: str):
    print(f"\n>>> Extracting faces for: {split_name.upper()}")

    config    = SPLIT_CONFIG[split_name]
    csv_path  = config["csv"]
    video_dir = config["video_dir"]
    output_path = PROCESSED_DIR / f"meld_{split_name}_faces.pt"

    df = pd.read_csv(csv_path, encoding='latin-1')

    # dev CSV uses 'joy' instead of 'happiness'
    if split_name == "dev":
        df['Emotion']   = df['Emotion'].replace({'joy': 'happiness'})
        df['Utterance'] = df['Utterance'].str.replace('\u2019', "'", regex=False)

    net = load_dnn_detector()

    extracted = []
    skipped   = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
        emotion = row['Emotion']
        if emotion not in EMOTION_MAPPING:
            skipped += 1
            continue

        v_filename = f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.mp4"
        v_path     = video_dir / v_filename

        if not v_path.exists():
            skipped += 1
            continue

        frame_tensors = extract_frames_from_video(v_path, net)

        if len(frame_tensors) == 0:
            skipped += 1
            continue

        extracted.append({
            'frames': torch.stack(frame_tensors),  # [T, 1, 48, 48]
            'label':  EMOTION_MAPPING[emotion],    # int
            'dia_id': int(row['Dialogue_ID']),     # kept for debugging
            'utt_id': int(row['Utterance_ID']),    # kept for debugging
        })

    torch.save(extracted, output_path)
    print(f"  Saved : {len(extracted)} utterances → {output_path}")
    print(f"  Skipped: {skipped} utterances (missing video or no face detected)")


if __name__ == "__main__":
    run_extraction("train")
    run_extraction("dev")
    run_extraction("test")