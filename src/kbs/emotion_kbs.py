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
    }

    def __init__(self, kb_path: str | Path | None = KBS_PATH / "emotion_kbs.pl"):
        self.prolog = Prolog()

        if kb_path is None:
            kb_path = Path(__file__).resolve().parent / "emotion_kb.pl"

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
            case             : str  (agreement / masking / conflict / partial / uncertain)
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

        context = f"[EMOTION ANALYSIS — KBS]\n"
        context += f"Detected emotion   : {emotion} ({description})\n"
        context += f"Confidence level   : {conf_level}\n"
        context += f"Fusion case        : {case}\n"
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

        explanations = {
            'agreement':  f"Both modalities agree → {dom}",
            'masking':    f"Emotional masking detected → trusting face ({dom})",
            'conflict':   f"Modalities conflict → higher confidence wins ({dom})",
            'partial':    f"Same polarity, different emotion → higher confidence ({dom})",
            'uncertain':  f"Low confidence in both → defaulting to neutral",
        }

        return explanations.get(case, f"Resolved to {dom}") + f" | Strategy: {strat}"