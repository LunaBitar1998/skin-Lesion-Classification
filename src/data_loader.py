import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from src.transforms import get_train_transform, get_val_transform


def get_data_loaders(train_dir, val_dir, img_size=224, batch_size=32, num_workers=4):
    # Define transforms
    train_transform = get_train_transform(size=img_size)
    val_transform = get_val_transform(size=img_size)

    # Create datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Log dataset info
    print(f"Training Dataset: {len(train_dataset)} images, {len(train_dataset.classes)} classes")
    print(f"Validation Dataset: {len(val_dataset)} images, {len(val_dataset.classes)} classes")

    return train_loader, val_loader
