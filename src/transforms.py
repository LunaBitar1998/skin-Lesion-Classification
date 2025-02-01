import torch
import numpy as np
import random
from PIL import Image
from torchvision import transforms

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#  Check if all four corners are dark
def are_corners_dark(img, threshold=100):
    """Return True if all four corners of the image are dark."""
    h, w, _ = img.shape
    corners = [
        img[:3, :3], img[:3, w-3:],  # Top-left, Top-right
        img[h-3:, :3], img[h-3:, w-3:]  # Bottom-left, Bottom-right
    ]
    return all(np.mean(corner) <= threshold for corner in corners)

# Crop dark borders & resize
def crop_and_resize(img, size, threshold=100):
    img = np.array(img)
    while are_corners_dark(img, threshold) and min(img.shape[:2]) > 5:
        img = img[2:-2, 2:-2]  # Crop 2 pixels from all sides
    img = Image.fromarray(img)
    img = transforms.CenterCrop(min(img.size))(img)  # Center crop to square
    return transforms.Resize((size, size))(img)  # Resize to target size

# Train data augmentation
def get_train_transform(size):
    return transforms.Compose([
        lambda img: crop_and_resize(img, size),  
        transforms.RandomHorizontalFlip(0.8),
        transforms.RandomVerticalFlip(0.8),
        transforms.RandomRotation(20, fill=0),
        transforms.RandomPerspective(0.2, 0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.6301, 0.4673, 0.4545], [0.1336, 0.1468, 0.1565])
    ])

#  Validation transforms (No Augmentation)
def get_val_transform(size):
    return transforms.Compose([
        lambda img: crop_and_resize(img, size), 
        transforms.ToTensor(),
        transforms.Normalize([0.6301, 0.4673, 0.4545], [0.1336, 0.1468, 0.1565])
    ])

# Set seed for consistency
set_seed(42)

