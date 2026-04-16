# src/kbs/emotion_kbs.py

from pyswip import Prolog
from pathlib import Path
import sys

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.config import KBS_PATH

class EmotionKBS:
    """
    Knowledge-Based System for multimodal emotion fusion.
    Uses SWI-Prolog via pyswip to reason over vision and text
    expert predictions and derive a response strategy.

    Fusion cases handled:
        agreement       — both experts predict the same emotion
        masking         — face negative, text neutral (emotional suppression)
        irony           — face negative, text positive (sarcasm/irony signal)
        neutral_override— one expert predicts neutral, trust the specific one
        partial         — same polarity, different specific emotion
        conflict        — opposite polarity, trust higher confidence
        uncertain       — both confidences too low to trust
    """

    EMOTION_DESCRIPTIONS = {
        'neutral':   'calm and neutral',
        'happiness': 'happy and positive',
        'surprise':  'surprised or caught off guard',
        'sadness':   'sad or down',
        'anger':     'angry or frustrated',
        'disgust':   'disgusted or uncomfortable',
        'fear':      'anxious or fearful',
    }

    STRATEGY_DESCRIPTIONS = {
        'empathetic_priority':   'strong negative emotion detected — prioritize empathy',
        'acknowledge_and_adapt': 'negative emotion detected — acknowledge and adapt tone',
        'reinforce_positive':    'positive emotion detected — reinforce and match energy',
        'gentle_acknowledgement':'ambiguous or mixed signals — be gently supportive',
        'neutral_supportive':    'neutral or uncertain — respond naturally and supportively',
        'irony_aware':            'irony or sarcasm detected — respond to underlying emotion'
                                  ' with subtle knowing awareness'
    }

    CASE_EXPLANATIONS = {
        'agreement':        "Both modalities agree",
        'masking':          "Emotional masking detected — face negative, words neutral or positive (sadness only)",
        'irony':            "Irony/sarcasm detected — face negative, words positive",
        'neutral_override': "Neutral overridden — trusting the specific prediction",
        'partial':          "Same emotional polarity, different specific emotion",
        'conflict':         "Modalities conflict — opposite polarity",
        'uncertain':        "Both experts have low confidence",
    }

    def __init__(self, kb_path: str | Path = KBS_PATH / "emotion_kbs.pl"):
        self.prolog = Prolog()
        self.prolog.consult(str(kb_path))

    def reason(
        self,
        vision_emotion: str,
        text_emotion:   str,
        vision_conf:    float,
        text_conf:      float,
    ) -> dict:
        """
        Query the Prolog KB with expert predictions.

        Returns a dict with:
            dominant_emotion : str
            confidence_level : str  (high / medium / low)
            strategy         : str
            case             : str  (agreement / masking / conflict / partial / uncertain / neutral_override / irony)
        """
        # Prolog atoms must be lowercase
        ve = vision_emotion.lower()
        te = text_emotion.lower()
        vc = round(float(vision_conf), 3)
        tc = round(float(text_conf),   3)

        query = (
            f"emotion_agent({ve}, {te}, {vc}, {tc}, "
            f"Strategy, DominantEmotion, ConfidenceLevel, Case)"
        )

        try:
            results = list(self.prolog.query(query))
        except Exception as e:
            print(f"[KBS] Prolog query failed: {e}")
            results = []

        if not results:
            return {
                'dominant_emotion': 'neutral',
                'confidence_level': 'low',
                'strategy':         'neutral_supportive',
                'case':             'uncertain',
            }

        r = results[0]
        return {
            'dominant_emotion': str(r['DominantEmotion']),
            'confidence_level': str(r['ConfidenceLevel']),
            'strategy':         str(r['Strategy']),
            'case':             str(r['Case']),
        }

    def build_llm_context(self, reasoning: dict) -> str:
        """
        Converts KBS reasoning output into a natural language
        string to inject into the LLM system prompt.
        """
        emotion     = reasoning['dominant_emotion']
        strategy    = reasoning['strategy']
        case        = reasoning['case']
        conf_level  = reasoning['confidence_level']

        description  = self.EMOTION_DESCRIPTIONS.get(emotion, 'unknown emotional state')
        strategy_desc = self.STRATEGY_DESCRIPTIONS.get(strategy, '')
        case_desc     = self.CASE_EXPLANATIONS.get(case, '')

        context = f"[EMOTION ANALYSIS — KBS]\n"
        context += f"Detected emotion   : {emotion} ({description})\n"
        context += f"Confidence level   : {conf_level}\n"
        context += f"Fusion case        : {case} - {case_desc}\n"
        context += f"Response strategy  : {strategy} — {strategy_desc}\n"

        return context

    def explain(self, reasoning: dict) -> str:
        """
        Returns a human-readable explanation of the KBS decision.
        Useful for display in the inference script overlay.
        """
        case = reasoning['case']
        dom  = reasoning['dominant_emotion']
        strat = reasoning['strategy']
        conf  = reasoning['confidence_level']

        case_desc = self.CASE_EXPLANATIONS.get(case, f"Resolved to {dom}")

        return f"{case_desc} → {dom} [{conf}] | {strat}"