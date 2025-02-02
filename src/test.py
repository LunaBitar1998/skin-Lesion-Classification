import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src.transforms import get_val_transform
from src.models import initialize_model
import src.config
import sys
import os

# ✅ Get the absolute path of the repo

#repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#sys.path.append(repo_path + "/src")  # ✅ Ensure src/ is in Python's path

# ✅ Import modules now
#from transforms import get_val_transform
#from models import initialize_model
#import config

def evaluate_model(model_name,dropout, test_dir, batch_size=32, device="cuda"):
    """Evaluate a trained model on a separate test set."""
    
    # Load the model
    model = initialize_model(model_name, dropout)
    model.load_state_dict(torch.load(f"/kaggle/working/{model_name}_final.pth", map_location=device))
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
    test_dir = "/kaggle/input/skinlesionbinary/val/val"  #here you should put your test set path 
    evaluate_model(model_name=config.MODEL_NAME, test_dir=test_dir)
