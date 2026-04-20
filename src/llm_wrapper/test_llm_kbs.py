import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.kbs.emotion_kbs import EmotionKBS
from src.llm_wrapper.llm_wrapper import LLMWrapper
from src.config import KBS_PROMPT_PATH

kbs = EmotionKBS()
llm = LLMWrapper(system_prompt_path=KBS_PROMPT_PATH)

scenarios = [
    ("sadness", "neutral",   0.72, 0.65, "I'm fine, don't worry about me."),
    ("anger",   "happiness", 0.60, 0.55, "Oh great, another Monday. Just what I needed."),
    ("fear",    "fear",      0.85, 0.80, "I have an important exam tomorrow and I can't stop thinking about it."),
    ("neutral", "sadness",   0.75, 0.60, "My dog passed away last week."),
]

for ve, te, vc, tc, message in scenarios:
    reasoning = kbs.reason(ve, te, vc, tc)
    context   = kbs.build_llm_context(reasoning)
    response  = llm.chat(message, context)
    print(f"\nCase : {reasoning['case']} | Strategy: {reasoning['strategy']}")
    print(f"User : {message}")
    print(f"LLM  : {response}")
    print("-" * 60)
    llm.reset()