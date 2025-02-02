# Configuration for Skin Lesion Classification
import torch
# Dataset paths (update paths to Kaggle dataset structure)
TRAIN_DIR = "/kaggle/input/skinlesionbinary/train/train"
VAL_DIR = "/kaggle/input/skinlesionbinary/val/val"

# Training parameters
MODEL_NAME = "efficientnet_b4"          # Change to "efficientnet_b0", etc., as needed
IMG_SIZE = 224                   # Image size for resizing
BATCH_SIZE =32                # Batch size
NUM_EPOCHS = 20                  # Number of training epochs
LEARNING_RATE = 1e-4             # Learning rate
WEIGHT_DECAY = 1e-4              # Weight decay for regularization
OPTIMIZER_NAME = "adam"          # Optimizer: "adam", "sgd"
LOSS_NAME = "bce"                # Loss function: "bce", "cross_entropy"
PATIENCE = 5                     # Patience for early stopping
DROPOUT= 0.5
# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Output paths
SAVE_PATH = f"{MODEL_NAME}_best.pth"  # Path to save the best model
METRICS_PATH = f"{MODEL_NAME}_metrics.png"  # Path to save training metrics
