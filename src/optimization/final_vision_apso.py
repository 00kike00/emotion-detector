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
from src.architectures.vision_net import VisionNet
from src.data_pipeline.loaders import get_fer_loaders
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

# 1. THE FITNESS FUNCTION
def evaluate_particle(params, train_loader, valid_loader):
    lr, dropout, hidden = params
    hidden = int(hidden)

    lr = 10 ** lr  # Convert back from log scale

    model = VisionNet(dropout_rate=dropout, hidden_units=hidden).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n  [Particle] lr={lr:.6f} | dropout={dropout:.3f} | hidden={hidden}")

    best_f1 = 0.0

    for epoch in range(PROXY_EPOCHS):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        model.eval()
        all_preds  = []
        all_labels = []
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        f1 = f1_score(all_labels, all_preds, average='macro') * 100
        best_f1 = max(best_f1, f1)
        print(f"    Epoch {epoch+1}/{PROXY_EPOCHS} | Loss: {avg_loss:.4f} | F1 Macro: {f1:.2f}%")

    del model, optimizer
    torch.cuda.empty_cache()

    return best_f1

# 2. MAIN EXECUTION
if __name__ == "__main__":
    set_seed(SEED)

    print(f"--- Starting APSO Hyperparameter Optimization on {DEVICE} ---")

    train_loader, valid_loader, _ = get_fer_loaders()

    bounds = (
        [-5, 0.2, 128],  # Min, we use logarmic scale for learning rate
        [-2, 0.5, 1024]  # Max, we use logarmic scale for learning rate
    )

    optimizer = APSO(
        fitness_function=lambda p: evaluate_particle(p, train_loader, valid_loader),
        num_particles=10,
        num_dimensions=3,
        bounds=bounds,
        max_iterations=10,
        alpha=0.3
    )

    best_config, best_score = optimizer.optimize()

    # 3. SAVE RESULTS
    results = {
        "best_learning_rate": float(best_config[0]),
        "best_dropout":       float(best_config[1]),
        "best_hidden_units":  int(best_config[2]),
        "best_f1_macro":      float(best_score)
    }

    with open(CHECKPOINTS_DIR / "final_vision_apso_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nOptimization Complete!")
    print(f"Best Config: {results}")