import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from src.config import DEVICE, GOEMOTIONS_DIR, CHECKPOINTS_DIR, PROXY_EPOCHS
from src.architectures.text_net import RobertaBiLSTM
from src.data_pipeline.loaders import get_text_loaders
from src.optimization.apso import APSO
import json
from time import sleep

POOLING_MODES = ['max', 'last', 'mean']

# 1. THE FITNESS FUNCTION
def evaluate_particle(params, train_loader, valid_loader):
    """
    This is what the APSO calls for every particle.
    params: [learning_rate, dropout_rate, hidden_units, pooling_mode]
    """
    lr, dropout, hidden, pooling_idx = params
    hidden = int(hidden)
    pooling_mode = POOLING_MODES[round(float(pooling_idx))]

    print(f"\n  [Particle] lr={lr:.6f} | dropout={dropout:.3f} | hidden={hidden} | pooling={pooling_mode}")

    # Initialize Model
    model = RobertaBiLSTM(
        dropout=dropout,
        hidden_dim=hidden,
        pooling_mode=pooling_mode
    ).to(DEVICE)

    # Freeze RoBERTa entirely
    for param in model.roberta.parameters():
        param.requires_grad = False

    # Only optimize BiLSTM + classifier
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    scaler = GradScaler('cuda')
    # Train for a few epochs to see potential
    for epoch in range(PROXY_EPOCHS):
        model.train()
        running_loss = 0.0
        # Training loop
        for batch in train_loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['label'].to(DEVICE)

            optimizer.zero_grad()

            # Enable Mixed Precision here
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

            # Use scaler for backward and step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # Validation Phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            # Validation loop
            for batch in valid_loader:
                input_ids      = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels         = batch['label'].to(DEVICE)

                # Use autocast in validation too for speed/temp consistency
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(input_ids, attention_mask)

                _, predicted = torch.max(outputs, 1)
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        best_val_acc = max(best_val_acc, val_acc)
        print(f"    Epoch {epoch+1}/{PROXY_EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

    print(f"  [Particle Done] Best Val Acc: {best_val_acc:.2f}%")

    # Clear VRAM for the next particle
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    sleep(2)

    return best_val_acc

# 2. MAIN EXECUTION
if __name__ == "__main__":
    print(f"--- Starting APSO Hyperparameter Optimization on {DEVICE} ---")

    # Load Data
    train_loader, valid_loader, _ = get_text_loaders()

    # Define Search Space: (Lower Bounds, Upper Bounds)
    # [Learning Rate, Dropout, Hidden Units, Pooling Mode]
    bounds = (
        [1e-5, 0.1,  64, 0],   # Min
        [1e-2, 0.5, 512, 2]    # Max
    )

    # Initialize Optimizer
    optimizer = APSO(
        fitness_function=lambda p: evaluate_particle(p, train_loader, valid_loader),
        num_particles=10,
        num_dimensions=4,
        bounds=bounds,
        max_iterations=10,
        alpha=0.3
    )

    # Run Search
    best_config, best_score = optimizer.optimize()

    # 3. SAVE RESULTS
    results = {
        "best_learning_rate": float(best_config[0]),
        "best_dropout":       float(best_config[1]),
        "best_hidden_units":  int(best_config[2]),
        "best_pooling_mode":  POOLING_MODES[round(float(best_config[3]))],
        "best_accuracy":      float(best_score)
    }

    with open(CHECKPOINTS_DIR / "text_apso_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nOptimization Complete!")
    print(f"Best Config: {results}")