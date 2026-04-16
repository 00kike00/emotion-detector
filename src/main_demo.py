import sys
import cv2
import torch
import numpy as np
import json
from pathlib import Path
from transformers import RobertaTokenizer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QTextEdit, QPushButton, QFrame, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QEvent
from PyQt6.QtGui import QImage, QPixmap, QFont

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.config import MODELS_DIR, CHECKPOINTS_DIR, DEVICE, KBS_PROMPT_PATH
from src.architectures.vision_net import VisionNet
from src.architectures.text_net import RobertaBiLSTM
from src.kbs.emotion_kbs import EmotionKBS
from src.llm_wrapper.llm_wrapper import LLMWrapper

# ── Constants ─────────────────────────────────────────────────────────────────
EMOTIONS = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
EMOTION_COLORS = {
    'neutral':   '#7c85a8',
    'happiness': '#f6c90e',
    'surprise':  '#ff9f43',
    'sadness':   '#54a0ff',
    'anger':     '#ee5a24',
    'disgust':   '#6ab04c',
    'fear':      '#a29bfe',
}
EMOTION_ICONS = {
    'neutral':   '😐', 'happiness': '😄', 'surprise': '😲',
    'sadness':   '😢', 'anger':     '😠', 'disgust':  '🤢', 'fear': '😨'
}
CASE_COLORS = {
    'agreement':       '#6ab04c',
    'masking':         '#54a0ff',
    'irony':           '#ff9f43',
    'neutral_override':'#7c85a8',
    'partial':         '#a29bfe',
    'conflict':        '#ee5a24',
    'uncertain':       '#636e72',
}

DNN_PROTOTXT   = Path("models/face_detection/deploy.prototxt")
DNN_WEIGHTS    = Path("models/face_detection/res10_300x300_ssd_iter_140000.caffemodel")
CONFIDENCE_THR = 0.5


# ── Model loading ─────────────────────────────────────────────────────────────
def load_models():
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

    ckpt_t      = torch.load(
        MODELS_DIR / "final_text_expert_best_ft.pth", map_location=DEVICE
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

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    dnn       = cv2.dnn.readNetFromCaffe(str(DNN_PROTOTXT), str(DNN_WEIGHTS))
    kbs       = EmotionKBS()
    llm       = LLMWrapper(system_prompt_path=KBS_PROMPT_PATH)

    return vision, text, tokenizer, dnn, kbs, llm


# ── Camera Thread ─────────────────────────────────────────────────────────────
class CameraThread(QThread):
    frame_ready   = pyqtSignal(QImage)
    emotion_ready = pyqtSignal(str, float, list)

    def __init__(self, vision_model, dnn_detector):
        super().__init__()
        self.vision_model = vision_model
        self.dnn          = dnn_detector
        self.running      = True

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0)
            )
            self.dnn.setInput(blob)
            detections = self.dnn.forward()

            best_conf, best_crop, best_box = 0.0, None, None
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf < CONFIDENCE_THR:
                    continue
                x1 = max(0, int(detections[0, 0, i, 3] * w))
                y1 = max(0, int(detections[0, 0, i, 4] * h))
                x2 = min(w, int(detections[0, 0, i, 5] * w))
                y2 = min(h, int(detections[0, 0, i, 6] * h))
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                if conf > best_conf:
                    best_conf = conf
                    best_crop = crop
                    best_box  = (x1, y1, x2, y2)

            if best_crop is not None and best_box is not None:
                x1, y1, x2, y2 = best_box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (92, 104, 152), 2)
                gray    = cv2.cvtColor(best_crop, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (48, 48))
                tensor  = torch.tensor(
                    resized, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0) / 255.0
                tensor  = ((tensor - 0.5) / 0.5).to(DEVICE)
                with torch.no_grad():
                    logits = self.vision_model(tensor)
                probs   = torch.softmax(logits.squeeze(), dim=-1).cpu().numpy()
                top_idx = int(np.argmax(probs))
                self.emotion_ready.emit(
                    EMOTIONS[top_idx], float(probs[top_idx]), probs.tolist()
                )

            rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.frame_ready.emit(qimg.copy())

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


# ── LLM Streaming Thread ──────────────────────────────────────────────────────
class LLMThread(QThread):
    token_ready  = pyqtSignal(str)
    stream_done  = pyqtSignal()

    def __init__(self, llm, user_input, emotion_context):
        super().__init__()
        self.llm             = llm
        self.user_input      = user_input
        self.emotion_context = emotion_context

    def run(self):
        for token in self.llm.stream(self.user_input, self.emotion_context):
            self.token_ready.emit(token)
        self.stream_done.emit()


# ── Emotion Bar Widget ────────────────────────────────────────────────────────
class EmotionBar(QWidget):
    def __init__(self, emotion: str, parent=None):
        super().__init__(parent)
        self.emotion = emotion
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(8)

        self.label = QLabel(emotion.capitalize())
        self.label.setFixedWidth(75)
        self.label.setStyleSheet("color: #7c85a8; font-size: 12px;")
        self.label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.bar_bg = QFrame()
        self.bar_bg.setFixedHeight(9)
        self.bar_bg.setStyleSheet("background: #12151f; border-radius: 4px;")

        self.bar_fill = QFrame(self.bar_bg)
        self.bar_fill.setFixedHeight(9)
        color = EMOTION_COLORS.get(emotion, '#5c6898')
        self.bar_fill.setStyleSheet(f"background: {color}; border-radius: 4px;")
        self.bar_fill.setFixedWidth(0)

        self.pct = QLabel("0%")
        self.pct.setFixedWidth(36)
        self.pct.setStyleSheet("color: #c9d1f5; font-size: 12px;")
        self.pct.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout.addWidget(self.label)
        layout.addWidget(self.bar_bg, 1)
        layout.addWidget(self.pct)

    def set_value(self, prob: float, bar_width: int):
        self.bar_fill.setFixedWidth(max(0, int(prob * bar_width)))
        self.pct.setText(f"{prob:.0%}")


# ── Main Window ───────────────────────────────────────────────────────────────
class EmotionChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multimodal Emotion Chat — KBS")
        self.setMinimumSize(1300, 820)
        self.setStyleSheet("background-color: #0f1117; color: #e0e0e0;")

        print("Loading models...")
        (self.vision_model, self.text_model,
         self.tokenizer, self.dnn, self.kbs, self.llm) = load_models()
        print("Models loaded.")

        self.current_face_emotion  = "neutral"
        self.current_face_conf     = 0.0
        self.current_face_probs    = [0.0] * 7
        self.current_text_emotion  = "neutral"
        self.current_text_conf     = 0.0
        self.current_text_probs    = [0.0] * 7
        self.current_final_emotion = "—"
        self.llm_thread            = None
        self.current_response      = ""

        self._build_ui()
        self._start_camera()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(14)

        # ── LEFT PANEL ────────────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(460)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        # Camera 
        self.camera_label = QLabel()
        self.camera_label.setFixedSize(460, 345)
        self.camera_label.setStyleSheet(
            "background: #1e2130; border: 1px solid #2e3250; border-radius: 10px;"
        )
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setText("Starting camera...")
        left_layout.addWidget(self.camera_label)

        # Emotion cards — wider, taller
        cards_widget = QWidget()
        cards_layout = QHBoxLayout(cards_widget)
        cards_layout.setContentsMargins(0, 0, 0, 0)
        cards_layout.setSpacing(8)
        self.face_card  = self._make_emotion_card("Face")
        self.text_card  = self._make_emotion_card("Text")
        self.final_card = self._make_emotion_card("Final")
        for card in [self.face_card, self.text_card, self.final_card]:
            card.setFixedHeight(95)
            cards_layout.addWidget(card)
        left_layout.addWidget(cards_widget)

        # Logit bars — face
        face_panel = self._make_panel("Face Distribution")
        self.face_bars = {}
        for e in EMOTIONS:
            bar = EmotionBar(e)
            self.face_bars[e] = bar
            face_panel.layout().addWidget(bar)
        left_layout.addWidget(face_panel)

        # Logit bars — text
        text_panel = self._make_panel("Text Distribution")
        self.text_bars = {}
        for e in EMOTIONS:
            bar = EmotionBar(e)
            self.text_bars[e] = bar
            text_panel.layout().addWidget(bar)
        left_layout.addWidget(text_panel)

        root_layout.addWidget(left)

        # ── RIGHT PANEL ───────────────────────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        # Title
        title = QLabel("Multimodal Emotion Chat")
        title.setStyleSheet("color: #c9d1f5; font-size: 20px; font-weight: 600;")
        right_layout.addWidget(title)

        subtitle = QLabel("")
        subtitle.setStyleSheet("color: #5c6898; font-size: 12px; margin-bottom: 4px;")
        right_layout.addWidget(subtitle)

        # 1. Chat history
        right_layout.addWidget(self._section_label("Conversation History"))
        self.chat_area = QScrollArea()
        self.chat_area.setWidgetResizable(True)
        self.chat_area.setMinimumHeight(220)
        self.chat_area.setStyleSheet(
            "QScrollArea { background: #13161f; border: 1px solid #2e3250;"
            " border-radius: 10px; }"
            "QScrollBar:vertical { background: #1e2130; width: 6px; border-radius: 3px; }"
            "QScrollBar::handle:vertical { background: #2e3250; border-radius: 3px; }"
        )
        self.chat_container = QWidget()
        self.chat_container.setStyleSheet("background: transparent;")
        self.chat_vbox = QVBoxLayout(self.chat_container)
        self.chat_vbox.setContentsMargins(10, 10, 10, 10)
        self.chat_vbox.setSpacing(8)
        self.chat_vbox.addStretch()
        self.chat_area.setWidget(self.chat_container)
        right_layout.addWidget(self.chat_area, 2)

        # 2. Your message
        right_layout.addWidget(self._section_label("Your Message"))
        self.input_box = QTextEdit()
        self.input_box.setFixedHeight(90)
        self.input_box.setStyleSheet(
            "QTextEdit {"
            "  background: #1e2130; border: 1px solid #2e3250;"
            "  border-radius: 10px; padding: 10px;"
            "  color: #e0e0e0; font-size: 14px;"
            "}"
            "QTextEdit:focus { border: 1px solid #5c6898; }"
        )
        self.input_box.setPlaceholderText("Type your message here...  (Enter to send, Shift+Enter for new line)")
        self.input_box.installEventFilter(self)
        right_layout.addWidget(self.input_box)

        # Send / Clear buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.send_btn = QPushButton("Send  ➤")
        self.send_btn.setFixedHeight(36)
        self.send_btn.setStyleSheet(
            "QPushButton { background: #5c6898; color: white; border: none;"
            " border-radius: 8px; font-size: 13px; font-weight: 500; padding: 0 20px; }"
            "QPushButton:hover { background: #4a567a; }"
            "QPushButton:disabled { background: #2e3250; color: #5c6898; }"
        )
        self.send_btn.clicked.connect(self._on_send)

        self.clear_btn = QPushButton("Clear History")
        self.clear_btn.setFixedHeight(36)
        self.clear_btn.setStyleSheet(
            "QPushButton { background: #1e2130; color: #7c85a8;"
            " border: 1px solid #2e3250; border-radius: 8px;"
            " font-size: 13px; padding: 0 16px; }"
            "QPushButton:hover { background: #2e3250; color: #c9d1f5; }"
        )
        self.clear_btn.clicked.connect(self._on_clear)
        btn_row.addWidget(self.send_btn)
        btn_row.addWidget(self.clear_btn)
        btn_row.addStretch()
        right_layout.addLayout(btn_row)

        # 3. KBS Reasoning panel
        right_layout.addWidget(self._section_label("KBS Reasoning"))
        kbs_frame = QFrame()
        kbs_frame.setStyleSheet(
            "QFrame { background: #1e2130; border: 1px solid #2e3250;"
            " border-radius: 10px; }"
        )
        kbs_inner = QVBoxLayout(kbs_frame)
        kbs_inner.setContentsMargins(14, 10, 14, 10)
        kbs_inner.setSpacing(5)

        self.kbs_case_label     = QLabel("Case     : —")
        self.kbs_strategy_label = QLabel("Strategy : —")
        self.kbs_explain_label  = QLabel("Explain  : —")
        self.kbs_explain_label.setWordWrap(True)

        for lbl in [self.kbs_case_label, self.kbs_strategy_label, self.kbs_explain_label]:
            lbl.setStyleSheet("color: #7c85a8; font-size: 13px; background: transparent; border: none;")
            kbs_inner.addWidget(lbl)

        right_layout.addWidget(kbs_frame)

        # 4. Assistant response
        right_layout.addWidget(self._section_label("Assistant"))
        self.response_box = QTextEdit()
        self.response_box.setReadOnly(True)
        self.response_box.setMinimumHeight(160)
        self.response_box.setStyleSheet(
            "QTextEdit {"
            "  background: #1e2130; border: 1px solid #2e3250;"
            "  border-radius: 10px; padding: 12px;"
            "  color: #d0d8f0; font-size: 14px; line-height: 1.6;"
            "}"
        )
        self.response_box.setPlaceholderText("Waiting for your message...")
        right_layout.addWidget(self.response_box, 2)

        root_layout.addWidget(right, 1)

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text.upper())
        lbl.setStyleSheet(
            "color: #5c6898; font-size: 11px; letter-spacing: 1px; font-weight: 600;"
        )
        return lbl

    def _make_panel(self, title: str) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { background: #1e2130; border: 1px solid #2e3250; border-radius: 10px; }"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        lbl = QLabel(title.upper())
        lbl.setStyleSheet(
            "color: #5c6898; font-size: 11px; letter-spacing: 1px;"
            "border: none; background: transparent;"
        )
        layout.addWidget(lbl)
        return frame

    def _make_emotion_card(self, label: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet(
            "QFrame { background: #1e2130; border: 1px solid #2e3250; border-radius: 10px; }"
        )
        layout = QVBoxLayout(card)
        layout.setContentsMargins(6, 8, 6, 8)
        layout.setSpacing(3)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_lbl = QLabel(label.upper())
        title_lbl.setStyleSheet(
            "color: #7c85a8; font-size: 11px; letter-spacing: 1px;"
            " border: none; background: transparent;"
        )
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_lbl = QLabel("❓")
        icon_lbl.setStyleSheet("font-size: 26px; border: none; background: transparent;")
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        value_lbl = QLabel("—")
        value_lbl.setStyleSheet(
            "color: #c9d1f5; font-size: 13px; font-weight: 600;"
            " border: none; background: transparent;"
        )
        value_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_lbl.setWordWrap(True)

        layout.addWidget(title_lbl)
        layout.addWidget(icon_lbl)
        layout.addWidget(value_lbl)

        card.icon_lbl  = icon_lbl
        card.value_lbl = value_lbl
        card.title_lbl = title_lbl
        return card

    def _update_emotion_card(self, card: QFrame, emotion: str, conf: float):
        icon  = EMOTION_ICONS.get(emotion, "❓")
        color = EMOTION_COLORS.get(emotion, "#c9d1f5")
        card.icon_lbl.setText(icon)
        card.value_lbl.setText(emotion.capitalize())
        card.value_lbl.setStyleSheet(
            f"color: {color}; font-size: 13px; font-weight: 600;"
            " border: none; background: transparent;"
        )

    # ── Camera ────────────────────────────────────────────────────────────────
    def _start_camera(self):
        self.cam_thread = CameraThread(self.vision_model, self.dnn)
        self.cam_thread.frame_ready.connect(self._on_frame)
        self.cam_thread.emotion_ready.connect(self._on_face_emotion)
        self.cam_thread.start()

    def _on_frame(self, qimg: QImage):
        pixmap = QPixmap.fromImage(qimg).scaled(
            460, 345,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.camera_label.setPixmap(pixmap)

    def _on_face_emotion(self, emotion: str, conf: float, probs: list):
        self.current_face_emotion = emotion
        self.current_face_conf    = conf
        self.current_face_probs   = probs
        self._update_emotion_card(self.face_card, emotion, conf)
        bar_w = self.face_bars[EMOTIONS[0]].bar_bg.width()
        for i, e in enumerate(EMOTIONS):
            self.face_bars[e].set_value(probs[i], bar_w)

    # ── Text inference ────────────────────────────────────────────────────────
    def _infer_text(self, text: str):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64
        )
        with torch.no_grad():
            logits = self.text_model(
                inputs['input_ids'].to(DEVICE),
                inputs['attention_mask'].to(DEVICE)
            )
        probs   = torch.softmax(logits.squeeze(), dim=-1).cpu().numpy()
        top_idx = int(np.argmax(probs))
        self.current_text_emotion = EMOTIONS[top_idx]
        self.current_text_conf    = float(probs[top_idx])
        self.current_text_probs   = probs.tolist()
        self._update_emotion_card(self.text_card, self.current_text_emotion, self.current_text_conf)
        bar_w = self.text_bars[EMOTIONS[0]].bar_bg.width()
        for i, e in enumerate(EMOTIONS):
            self.text_bars[e].set_value(probs[i], bar_w)

    # ── Send ──────────────────────────────────────────────────────────────────
    def _on_send(self):
        user_text = self.input_box.toPlainText().strip()
        if not user_text:
            return
        self.input_box.clear()
        self.send_btn.setEnabled(False)

        self._infer_text(user_text)

        reasoning = self.kbs.reason(
            self.current_face_emotion,
            self.current_text_emotion,
            self.current_face_conf,
            self.current_text_conf
        )

        self.current_final_emotion = reasoning['dominant_emotion']
        self._update_emotion_card(self.final_card, self.current_final_emotion, self.current_face_conf)

        # KBS panel update
        color = CASE_COLORS.get(reasoning['case'], '#7c85a8')
        self.kbs_case_label.setText(f"Case     : {reasoning['case']}")
        self.kbs_strategy_label.setText(f"Strategy : {reasoning['strategy']}")
        self.kbs_explain_label.setText(f"Explain  : {self.kbs.explain(reasoning)}")
        for lbl in [self.kbs_case_label, self.kbs_strategy_label, self.kbs_explain_label]:
            lbl.setStyleSheet(
                f"color: {color}; font-size: 13px; background: transparent; border: none;"
            )

        self._add_chat_message("You", user_text, "#2a3350")

        emotion_context = self.kbs.build_llm_context(reasoning)
        self.current_response = ""
        self.response_box.clear()

        self.llm_thread = LLMThread(self.llm, user_text, emotion_context)
        self.llm_thread.token_ready.connect(self._on_token)
        self.llm_thread.stream_done.connect(self._on_stream_done)
        self.llm_thread.start()

    def _on_token(self, token: str):
        self.current_response += token
        self.response_box.setPlainText(self.current_response)
        self.response_box.verticalScrollBar().setValue(
            self.response_box.verticalScrollBar().maximum()
        )

    def _on_stream_done(self):
        self._add_chat_message("Assistant", self.current_response, "#1a2240")
        self.send_btn.setEnabled(True)

    # ── Chat history ──────────────────────────────────────────────────────────
    def _add_chat_message(self, role: str, text: str, bg_color: str):
        # Keep max 4 bubbles (2 exchanges)
        while self.chat_vbox.count() > 5:
            item = self.chat_vbox.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        bubble = QFrame()
        bubble.setStyleSheet(
            f"QFrame {{ background: {bg_color}; border-radius: 10px; }}"
        )
        b_layout = QVBoxLayout(bubble)
        b_layout.setContentsMargins(12, 8, 12, 8)
        b_layout.setSpacing(3)

        role_color = "#8899cc" if role == "You" else "#6ab04c"
        role_lbl = QLabel(role)
        role_lbl.setStyleSheet(
            f"color: {role_color}; font-size: 11px; font-weight: 600;"
            " background: transparent; border: none;"
        )

        text_lbl = QLabel(text)
        text_lbl.setWordWrap(True)
        text_lbl.setStyleSheet(
            "color: #d0d8f0; font-size: 13px; background: transparent; border: none;"
        )

        b_layout.addWidget(role_lbl)
        b_layout.addWidget(text_lbl)

        self.chat_vbox.insertWidget(self.chat_vbox.count() - 1, bubble)
        QTimer.singleShot(50, lambda: self.chat_area.verticalScrollBar().setValue(
            self.chat_area.verticalScrollBar().maximum()
        ))

    # ── Clear ─────────────────────────────────────────────────────────────────
    def _on_clear(self):
        self.llm.reset()
        self.response_box.clear()
        self.response_box.setPlaceholderText("Waiting for your message...")
        self.current_response = ""
        while self.chat_vbox.count() > 1:
            item = self.chat_vbox.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for lbl in [self.kbs_case_label, self.kbs_strategy_label, self.kbs_explain_label]:
            lbl.setText(lbl.text().split(":")[0] + ": —")
            lbl.setStyleSheet("color: #7c85a8; font-size: 13px; background: transparent; border: none;")

    # ── Enter to send ─────────────────────────────────────────────────────────
    def eventFilter(self, obj, event):
        if obj is self.input_box and event.type() == QEvent.Type.KeyPress:
            if (event.key() == Qt.Key.Key_Return and
                    not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)):
                self._on_send()
                return True
        return super().eventFilter(obj, event)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        self.cam_thread.stop()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = EmotionChatWindow()
    window.show()
    sys.exit(app.exec())