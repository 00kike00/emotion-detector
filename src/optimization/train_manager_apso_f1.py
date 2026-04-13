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
from src.config import DEVICE, CHECKPOINTS_DIR, PROXY_EPOCHS
from src.architectures.manager_net import ManagerNet
from src.data_pipeline.loaders import get_meld_loaders
from src.optimization.apso import APSO
import json

SEED = 42

# hidden_dim is continuous in APSO but must snap to valid discrete values
HIDDEN_DIM_OPTIONS = [64, 128, 256, 512]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_particle(params, train_loader, dev_loader):
    lr, dropout, hidden_dim_cont, num_layers_cont = params

    hidden_dim = int(round(np.clip(hidden_dim_cont, 64, 512)))
    num_layers = int(round(np.clip(num_layers_cont, 1, 3)))

    print(f"\n  [Particle] lr={lr:.6f} | dropout={dropout:.3f} | "
          f"hidden_dim={hidden_dim} | num_layers={num_layers}")

    model = ManagerNet(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout_rate=dropout,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_f1 = 0.0

    for epoch in range(PROXY_EPOCHS):
        model.train()
        running_loss = 0.0

        for vision_seq, lengths, text_logits, labels in train_loader:
            vision_seq  = vision_seq.to(DEVICE)
            text_logits = text_logits.to(DEVICE)
            labels      = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(vision_seq, text_logits)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Validation
        model.eval()
        all_preds  = []
        all_labels = []
        with torch.no_grad():
            for vision_seq, lengths, text_logits, labels in dev_loader:
                vision_seq  = vision_seq.to(DEVICE)
                text_logits = text_logits.to(DEVICE)

                outputs = model(vision_seq, text_logits)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

        f1 = f1_score(all_labels, all_preds, average='macro') * 100
        best_f1 = max(best_f1, f1)
        print(f"    Epoch {epoch+1}/{PROXY_EPOCHS} | Loss: {avg_loss:.4f} | F1 Macro: {f1:.2f}%")

    del model, optimizer
    torch.cuda.empty_cache()

    return best_f1


if __name__ == "__main__":
    set_seed(SEED)

    print(f"--- Starting APSO Manager Optimization on {DEVICE} ---")

    train_loader, dev_loader, _ = get_meld_loaders()

    # [lr, dropout, hidden_dim, num_layers]
    bounds = (
        [1e-5, 0.2,  64, 1.0],  # Min
        [1e-2, 0.5, 512, 3.0]   # Max
    )

    optimizer = APSO(
        fitness_function=lambda p: evaluate_particle(p, train_loader, dev_loader),
        num_particles=10,
        num_dimensions=4,
        bounds=bounds,
        max_iterations=10,
        alpha=0.3
    )

    best_config, best_score = optimizer.optimize()

    results = {
        "best_learning_rate": float(best_config[0]),
        "best_dropout":       float(best_config[1]),
        "best_hidden_dim":    int(round(np.clip(best_config[2], 64, 512))),
        "best_num_layers":    int(round(np.clip(best_config[3], 1, 3))),
        "best_f1_macro":      float(best_score)
    }

    with open(CHECKPOINTS_DIR / "manager_apso_f1_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nOptimization Complete!")
    print(f"Best Config: {results}")