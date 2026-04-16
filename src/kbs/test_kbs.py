import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.kbs.emotion_kbs import EmotionKBS

kbs = EmotionKBS()

test_cases = [
    # (vision, text, v_conf, t_conf, description)
    ("happiness", "happiness", 0.80, 0.75, "Agreement — both happy"),
    ("sadness",   "neutral",   0.72, 0.65, "Masking — face sad, text neutral"),
    ("anger",     "happiness", 0.60, 0.55, "Irony — anger vs happiness"),
    ("sadness",   "fear",      0.58, 0.70, "Partial — both negative"),
    ("neutral",   "neutral",   0.30, 0.28, "Uncertain — both low confidence"),
    ("fear",      "fear",      0.85, 0.80, "Agreement — high intensity"),
    ("neutral",  "sadness",   0.75, 0.60, "Neutral override — vision neutral, text sad"),
    ("neutral",  "happiness", 0.80, 0.55, "Neutral override — vision neutral, text happy"),
    ("anger",    "neutral",   0.65, 0.70, "Neutral override — vision angry, text neutral"),
    ("disgust",  "happiness", 0.75, 0.40, "Irony — disgust face, happy words, high intensity"),
    ("sadness",   "happiness", 0.70, 0.65, "Masking — sad face, happy words"),
    ("happiness", "anger",     0.70, 0.65, "Conflict — happy face, angry words"),
    ("surprise",  "sadness",   0.55, 0.80, "Conflict — surprised face, sad words, text more confident"),
    ("sadness",   "surprise",  0.65, 0.60, "Masking — sad face, surprised words (not irony)"),
    ("neutral",   "anger",     0.60, 0.55, "Neutral override — vision neutral, text angry"),
]

print("=" * 60)
print("KBS TEST CASES")
print("=" * 60)

for ve, te, vc, tc, desc in test_cases:
    result = kbs.reason(ve, te, vc, tc)
    explain = kbs.explain(result)
    print(f"\n{desc}")
    print(f"  Input    : vision={ve}({vc:.0%})  text={te}({tc:.0%})")
    print(f"  Decision : {explain}")
    print(f"  Strategy : {result['strategy']}")
    print(f"  Context  : {kbs.build_llm_context(result)}")