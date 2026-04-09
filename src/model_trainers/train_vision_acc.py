import torch
import torch.nn as nn
import torch.optim as optim
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Path Fix
root = Path(__file__).resolve().parent.parent.parent
if str(root) not in sys.path:
    sys.path.append(str(root))

from src.config import DEVICE, CHECKPOINTS_DIR, MODELS_DIR, BATCH_SIZE, PLOTS_DIR
from src.architectures.vision_net import VisionNet
from src.data_pipeline.loaders import get_fer_loaders

def train_final():
    # 1. Load APSO Results
    with open(CHECKPOINTS_DIR / "vision_apso_results.json", "r") as f:
        best_params = json.load(f)
    
    print(f"--- Training Final Vision Expert ---")
    print(f"Parameters: {best_params}")

    history = {
        'train_loss': [],
        'val_acc': []
    }
    # 2. Setup Data & Model
    train_loader, valid_loader, test_loader = get_fer_loaders()
    model = VisionNet(
        dropout_rate=best_params['best_dropout'], 
        hidden_units=best_params['best_hidden_units']
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['best_learning_rate'])
    
    # Scheduler: Reduces LR if accuracy plateaus to "fine-tune" the weights
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)

    # 3. Training Loop
    epochs = 100
    best_acc = 0.0
    patience_counter = 0
    early_stop_patience = 12

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total

        history['train_loss'].append(train_loss / len(train_loader))
        history['val_acc'].append(val_acc)

        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODELS_DIR / "vision_expert_best.pth")
            patience_counter = 0
            print(f"  [SAVED] New best accuracy: {best_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Early Stopping
        if patience_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break
    # 4 Final Test Evaluation
    print("\n--- Running Final Test Evaluation ---")
    # Load the best weights we saved during the loop
    model.load_state_dict(torch.load(MODELS_DIR / "vision_expert_best.pth"))
    model.eval()
    
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    final_test_acc = 100 * test_correct / test_total
    print(f"Final Test Accuracy: {final_test_acc:.2f}%")

    # 5. Generate and Save Plot
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Loss', color='royalblue')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Acc', color='seagreen')
    plt.axhline(y=final_test_acc, color='r', linestyle='--', label=f'Test Acc: {final_test_acc:.1f}%')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    print(f"Final Training Complete. Best Val Acc: {best_acc:.2f}%")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "vision_training_curves.png")
    plt.show()

if __name__ == "__main__":
    train_final()