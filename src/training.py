import torch
import copy
from tqdm import tqdm
from src.early_stopping import early_stopping  

def train_model(
    model, model_name, train_loader, val_loader, criterion, optimizer, 
    scheduler=None, num_epochs=30, patience=5, device="cuda"
):
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc, counter = 0.0, 0

    best_model_path = f"{model_name}_best.pth"  # Save best model using model name
    final_model_path = f"{model_name}_final.pth"  # Save final model

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training phase
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

        # Validation phase
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

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if scheduler:
            scheduler.step(val_acc) if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else scheduler.step()

        # Save the best model with model-specific name
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, best_model_path)
            print(f"Best model updated! Saved to {best_model_path}")

        best_val_acc, counter, stop_training = early_stopping(val_acc, best_val_acc, counter, patience)
        if stop_training:
            print("Early stopping triggered!")
            break

    # Load best model weights & save final best model
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Final best model saved to {final_model_path}")

    return model
