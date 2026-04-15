import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from src.config import DEVICE, CHECKPOINTS_DIR
from src.architectures.manager_net import ManagerNet
from src.data_pipeline.loaders import get_meld_loaders
from src.optimization.apso import APSO
import json

SEED = 42

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

PROXY_EPOCHS = 8
# 1. FITNESS FUNCTION
def evaluate_particle(params, train_loader, dev_loader):
    lr, dropout, hidden, num_layers = params

    lr         = 10 ** lr       # log scale → real lr
    hidden     = int(hidden)
    num_layers = int(round(num_layers))

    print(f"\n  [Particle] lr={lr:.6f} | dropout={dropout:.3f} | "
          f"hidden={hidden} | num_layers={num_layers}")

    model = ManagerNet(
        num_classes=7,
        vision_input_dim=7,
        text_input_dim=7,
        hidden_dim=hidden,
        num_layers=num_layers,
        dropout_rate=dropout,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_f1 = 0.0

    for epoch in range(PROXY_EPOCHS):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0

        for vision_seq, lengths, text_logits, labels in train_loader:
            vision_seq  = vision_seq.to(DEVICE)   # [B, T_max, 7]
            text_logits = text_logits.to(DEVICE)   # [B, 7]
            labels      = labels.to(DEVICE)        # [B]

            optimizer.zero_grad()
            outputs = model(vision_seq, text_logits)  # [B, 7]
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        all_preds  = []
        all_labels = []

        with torch.no_grad():
            for vision_seq, lengths, text_logits, labels in dev_loader:
                vision_seq  = vision_seq.to(DEVICE)
                text_logits = text_logits.to(DEVICE)
                labels      = labels.to(DEVICE)

                outputs = model(vision_seq, text_logits)
                preds   = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100
        print(f"    Epoch {epoch+1}/{PROXY_EPOCHS} | "
              f"Loss: {avg_loss:.4f} | Macro F1: {f1:.2f}%")

    best_f1 = max(best_f1, f1)  # Get the last F1 as the metric to know if the hyperparameters are good
    del model, optimizer
    torch.cuda.empty_cache()

    return best_f1


# 2. MAIN EXECUTION
if __name__ == "__main__":
    set_seed(SEED)

    print(f"--- Starting APSO Manager Hyperparameter Optimization on {DEVICE} ---")

    train_loader, dev_loader, _ = get_meld_loaders()

    # Search space:
    # lr         : log scale [-5, -2]  → real [1e-5, 1e-2]
    # dropout    : [0.1, 0.5]
    # hidden_dim : [64, 256]
    # num_layers : [1, 2]              → rounded to nearest int
    bounds = (
        [-5,  0.1,  64, 1.0],   # min
        [-2,  0.5, 256, 2.0]    # max
    )

    optimizer = APSO(
        fitness_function=lambda p: evaluate_particle(p, train_loader, dev_loader),
        num_particles=15,
        num_dimensions=4,
        bounds=bounds,
        max_iterations=15,
        alpha=0.3
    )

    best_config, best_score = optimizer.optimize()

    # 3. SAVE RESULTS
    results = {
        "best_learning_rate": float(best_config[0]),
        "best_dropout":       float(best_config[1]),
        "best_hidden_units":  int(best_config[2]),
        "best_num_layers":    int(round(best_config[3])),
        "best_f1_macro":      float(best_score)
    }

    with open(CHECKPOINTS_DIR / "final_manager_apso_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nOptimization Complete!")
    print(f"Best Config: {results}")