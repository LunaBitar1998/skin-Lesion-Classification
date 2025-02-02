import torch
import copy
import os
from tqdm import tqdm
from src.utils import early_stopping, plot_metrics  
from IPython.display import display, FileLink  # ✅ Added for generating download links

def train_model(
    model, model_name, train_loader, val_loader, optimizer, criterion, 
    scheduler=None, num_epochs=30, patience=5, device="cuda", resume=False
):
    best_model_path = f"{model_name}_best.pth"
    final_model_path = f"{model_name}_final.pth"
    
    model = model.to(device)

    best_val_acc, counter, start_epoch = 0.0, 0, 0
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    # Resume from the best checkpoint if available
    if resume and os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"]) if scheduler else None
        best_val_acc = checkpoint["best_val_acc"]
        start_epoch = checkpoint["epoch"] + 1
        counter = checkpoint["counter"]
        print(f"Resuming training from epoch {start_epoch}, best val accuracy: {best_val_acc:.4f}")

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1).float()
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss /= total_train
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1).float()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).long()
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= total_val
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if scheduler:
            scheduler.step(val_acc) if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else scheduler.step()

        # 🔹 Call early stopping FIRST, before updating best_val_acc
        best_val_acc, counter, stop_training = early_stopping(val_acc, best_val_acc, counter, patience, mode="max")

        if val_acc > best_val_acc:  # ✅ Only update AFTER early stopping check
            best_model_wts = copy.deepcopy(model.state_dict())
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": best_model_wts,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_val_acc": best_val_acc,  # ✅ Now updates after early stopping check
                "counter": counter
            }
            torch.save(checkpoint, best_model_path)
            print(f"Best model updated! Saved to {best_model_path}")

        if stop_training:
            print("Early stopping triggered!")
            break  # ✅ Now breaks at the right time

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Final best model saved to {final_model_path}")

    # Plot training metrics
    metrics_path = f"{model_name}_metrics.png"
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, metrics_path)
    print(f"Training metrics saved to {metrics_path}")

    # ✅ Add a download link for the final saved model
    print(f"Generating download link for final model: {final_model_path}")
    display(FileLink(final_model_path))  # Generate a clickable download link

    return model

