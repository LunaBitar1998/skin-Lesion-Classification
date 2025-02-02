import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from transforms import get_val_transform
from models import initialize_model
import config
import sys
import os

def evaluate_model(model_name, dropout, test_dir, batch_size=32, device="cuda"):
    """Evaluate a trained model on a separate test set."""
    
    # Load the model
    model = initialize_model(model_name, dropout)
    model_path = f"/kaggle/working/Skin-Lesion-Classification/{model_name}_best.pth"  # Use the best model
    checkpoint = torch.load(model_path, map_location=device)  # Load the checkpoint
    model.load_state_dict(checkpoint["model_state_dict"])  # Extract model weights
    model.to(device)
    model.eval()

    # Prepare test dataset
    test_transform = get_val_transform(config.IMG_SIZE)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Evaluation metrics
    correct, total = 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).long()  # Convert logits to binary labels
            
            correct += (preds == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute accuracy
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")

    return all_preds, all_labels, accuracy

if __name__ == "__main__":
    test_dir = "/kaggle/input/skinlesionbinary/val/val"  # Specify your test dataset path
    evaluate_model(
        model_name=config.MODEL_NAME,
        dropout=config.DROPOUT,
        test_dir=test_dir
    )
