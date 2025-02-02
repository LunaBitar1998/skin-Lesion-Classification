import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from torchvision import datasets
from models import initialize_model  
from transforms import get_val_transform  
from config import DROPOUT, BATCH_SIZE, IMG_SIZE

# 📌 Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📌 Model paths
model_paths = [
    "/kaggle/working/Skin-Lesion-Classification/efficientnet_b4_best.pth",
    "/kaggle/working/Skin-Lesion-Classification/densenet_121_best.pth",
    "/kaggle/working/Skin-Lesion-Classification/convnext_tiny_best.pth"
]

# 📌 Load all models correctly
def load_models(model_paths, device):
    models = []
    for model_path in model_paths:
        # ✅ Extract model name dynamically
        model_name = os.path.basename(model_path).replace("_best.pth", "")

        # ✅ Initialize model correctly with config dropout
        model = initialize_model(model_name, DROPOUT)  

        # ✅ Load weights correctly
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:  
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model.to(device)
        model.eval()
        models.append(model)

    return models

# 📌 Function to evaluate ensemble
def ensemble_predict(model_paths, test_dir, method="majority"):
    print(f"\n🔹 Evaluating ensemble using method: {method}")
    models = load_models(model_paths, device)

    # ✅ Keep using validation transform
    test_transform = get_val_transform(IMG_SIZE)  

    # 📌 Load dataset
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
                probs = torch.sigmoid(outputs)  
                predictions.append(probs.cpu().numpy())

            # 📌 Apply ensemble method
            predictions = np.array(predictions)  # Convert list to numpy array
            if method == "majority":
                final_preds = np.round(np.mean(predictions, axis=0))  
            elif method == "average":
                final_preds = np.mean(predictions, axis=0)  
            elif method == "max_prob":
                final_preds = np.argmax(np.sum(predictions, axis=0), axis=1)  
            else:
                raise ValueError("Unknown ensemble method")

            all_predictions.extend(final_preds.flatten())  # Flatten to match targets

    # 📌 Compute accuracy
    accuracy = np.mean(np.array(all_predictions) == np.array(all_targets))
    print(f"✅ Ensemble Accuracy ({method}): {accuracy:.4f}")

# 📌 Run ensemble
if __name__ == "__main__":
    test_dir = "/kaggle/input/skinlesionbinary/val/val"  # ✅ Correct test dataset path
    ensemble_method = "majority"  # Change to "average" or "max_prob" if needed
    ensemble_predict(model_paths, test_dir, method=ensemble_method)
