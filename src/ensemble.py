import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from transforms import get_val_transform
from models import initialize_model
import config as config

def ensemble_predict(
    model_paths, test_dir, method="majority", batch_size=32, device="cuda"
):
    """
    Perform ensemble predictions on the test dataset.

    Args:
        model_paths (list): List of paths to the saved model weights.
        test_dir (str): Path to the test dataset.
        method (str): Ensemble method ("majority", "average", "highest").
        batch_size (int): Batch size for DataLoader.
        device (str): Device to run the models on.

    Returns:
        Tuple: (accuracy, predictions, labels)
    """
    # Prepare the test dataset and DataLoader
    test_transform = get_val_transform(config.IMG_SIZE)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Load all models
    models = []
    for model_path in model_paths:
        model = initialize_model(config.MODEL_NAME, config.DROPOUT)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        models.append(model)

    # Perform ensemble predictions
    all_preds, all_labels = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Collect outputs from all models
            outputs_list = [torch.sigmoid(model(inputs)) for model in models]  # Sigmoid for binary classification

            if method == "majority":
                # Majority voting
                preds_list = [(outputs > 0.5).long() for outputs in outputs_list]
                stacked_preds = torch.stack(preds_list, dim=0)  # Shape: (num_models, batch_size, 1)
                majority_preds = torch.mode(stacked_preds, dim=0).values.squeeze(1)
                final_preds = majority_preds

            elif method == "average":
                # Averaging probabilities
                avg_probs = torch.mean(torch.stack(outputs_list, dim=0), dim=0)  # Shape: (batch_size, 1)
                final_preds = (avg_probs > 0.5).long()

            elif method == "highest":
                # Highest overall probability
                max_probs, _ = torch.max(torch.stack(outputs_list, dim=0), dim=0)  # Shape: (batch_size, 1)
                final_preds = (max_probs > 0.5).long()

            else:
                raise ValueError(f"Unknown ensemble method: {method}")

            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (final_preds == labels.unsqueeze(1)).sum().item()
            total += labels.size(0)

    # Compute accuracy
    accuracy = correct / total
    print(f"Ensemble ({method}) Accuracy: {accuracy:.4f} ({correct}/{total})")

    return accuracy, all_preds, all_labels


if __name__ == "__main__":
    test_dir = "/kaggle/input/skinlesionbinary/val/val"  # Path to your test dataset

    # List of trained models
    model_paths = [
        "/kaggle/working/Skin-Lesion-Classification/efficientnet_b4_best.pth",
        "/kaggle/working/Skin-Lesion-Classification/densenet_121_best.pth",
        "/kaggle/working/Skin-Lesion-Classification/convnext_tiny_best.pth"
    ]

    # Choose ensemble method
    for method in ["majority", "average", "highest"]:
        print(f"\n Evaluating ensemble using method: {method}")
        ensemble_predict(model_paths, test_dir, method=method)
