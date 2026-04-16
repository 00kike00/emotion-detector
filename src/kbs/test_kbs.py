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
    ("anger",     "happiness", 0.60, 0.55, "Conflict — anger vs happiness"),
    ("sadness",   "fear",      0.58, 0.70, "Partial — both negative"),
    ("neutral",   "neutral",   0.30, 0.28, "Uncertain — both low confidence"),
    ("fear",      "fear",      0.85, 0.80, "Agreement — high intensity"),
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