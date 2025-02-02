import torch
import torch.nn.functional as F
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import CustomDataset 
from models import initialize_model  

# 📌 Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📌 Model paths - Make sure these are correctly trained models!
model_paths = [
    "/kaggle/working/Skin-Lesion-Classification/efficientnet_b4_best.pth",
    "/kaggle/working/Skin-Lesion-Classification/densenet_121_best.pth",
    "/kaggle/working/Skin-Lesion-Classification/convnext_tiny_best.pth"
]

# 📌 Load all models correctly
def load_models(model_paths, device):
    models = []
    for model_path in model_paths:
        if "efficientnet_b4" in model_path:
            model_name = "efficientnet_b4"
        elif "densenet_121" in model_path:
            model_name = "densenet_121"
        elif "convnext_tiny" in model_path:
            model_name = "convnext_tiny"
        else:
            raise ValueError(f"Unknown model in {model_path}")

        # ✅ Initialize model correctly
        model = initialize_model(model_name, dropout=0.2)  # Adjust dropout if needed
        
        # ✅ Load checkpoint correctly
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)  # Allow partial match

        model.to(device)
        model.eval()
        models.append(model)

    return models

# 📌 Ensemble methods
def majority_voting(predictions):
    """Majority vote among models."""
    preds = np.array(predictions)
    final_preds = np.round(preds.mean(axis=0))  # Majority Voting
    return final_preds

def average_ensemble(predictions):
    """Averaging the softmax probabilities."""
    preds = np.array(predictions)
    return preds.mean(axis=0)  # Average of probabilities

def max_probability(predictions):
    """Select the class with the highest probability sum."""
    preds = np.array(predictions)
    return np.argmax(preds.sum(axis=0), axis=1)  # Highest summed probability

# 📌 Function to evaluate ensemble
def ensemble_predict(model_paths, test_dir, method="majority"):
    print(f"\n🔹 Evaluating ensemble using method: {method}")
    models = load_models(model_paths, device)

    # 📌 Define test transformations
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 📌 Load dataset
    test_dataset = CustomDataset(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    all_predictions = []
    all_targets = []

    # 📌 Iterate over test set
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            all_targets.extend(targets.cpu().numpy())

            predictions = []
            for model in models:
                outputs = model(inputs)
                probs = torch.sigmoid(outputs)  # Sigmoid for binary classification
                predictions.append(probs.cpu().numpy())

            # 📌 Apply ensemble method
            if method == "majority":
                final_preds = majority_voting(predictions)
            elif method == "average":
                final_preds = average_ensemble(predictions)
            elif method == "max_prob":
                final_preds = max_probability(predictions)
            else:
                raise ValueError("Unknown ensemble method")

            all_predictions.extend(final_preds)

    # 📌 Compute accuracy
    accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
    print(f"✅ Ensemble Accuracy ({method}): {accuracy:.4f}")

# 📌 Run ensemble
if __name__ == "__main__":
    test_dir = "/kaggle/input/skin-lesion-test"
    ensemble_method = "majority"  # Change to "average" or "max_prob" to try different ensembling
    ensemble_predict(model_paths, test_dir, method=ensemble_method)
