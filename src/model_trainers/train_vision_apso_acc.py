import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

import torch
import torch.nn as nn
import torch.optim as optim
from src.config import DEVICE, FER_DIR, CHECKPOINTS_DIR, BATCH_SIZE, PROXY_EPOCHS
from src.architectures.vision_net import VisionNet
from src.data_pipeline.loaders import get_fer_loaders
from src.optimization.apso import APSO
import json

# 1. THE FITNESS FUNCTION
def evaluate_particle(params, train_loader, valid_loader):
    """
    This is what the APSO calls for every particle.
    params: [learning_rate, dropout_rate, hidden_units]
    """
    lr, dropout, hidden = params
    hidden = int(hidden) # Ensure units are an integer
    
    print(f"\n  [Particle] lr={lr:.6f} | dropout={dropout:.3f} | hidden={hidden}")

    # Initialize Model
    model = VisionNet(dropout_rate=dropout, hidden_units=hidden).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0
    
    # Train for a few epochs to see potential
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
        # Validation Phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        best_val_acc = max(best_val_acc, val_acc)
        print(f"    Epoch {epoch+1}/{PROXY_EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")
    # Clear VRAM for the next particle
    del model, optimizer, train_loader, valid_loader
    torch.cuda.empty_cache()
    
    return best_val_acc

# 2. MAIN EXECUTION
if __name__ == "__main__":
    print(f"--- Starting APSO Hyperparameter Optimization on {DEVICE} ---")
    
    # Load Data
    train_loader, valid_loader, _ = get_fer_loaders()
    
    # Define Search Space: (Lower Bounds, Upper Bounds)
    # [Learning Rate, Dropout, Hidden Units]
    bounds = (
        [1e-5, 0.2, 128],  # Min
        [1e-2, 0.6, 1024]  # Max
    )
    
    # Initialize Optimizer
    # Using 10 particles for 10 iterations
    optimizer = APSO(
        fitness_function=lambda p: evaluate_particle(p, train_loader, valid_loader),
        num_particles=10,
        num_dimensions=3,
        bounds=bounds,
        max_iterations=10,
        alpha=0.3
    )
    
    # Run Search
    best_config, best_score = optimizer.optimize()
    
    # 3. SAVE RESULTS
    results = {
        "best_learning_rate": float(best_config[0]),
        "best_dropout": float(best_config[1]),
        "best_hidden_units": int(best_config[2]),
        "best_accuracy": float(best_score)
    }
    
    with open(CHECKPOINTS_DIR / "vision_apso_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nOptimization Complete!")
    print(f"Best Config: {results}")