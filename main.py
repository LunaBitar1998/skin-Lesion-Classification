import torch
from src.config import (
    TRAIN_DIR, VAL_DIR, MODEL_NAME, IMG_SIZE, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
    WEIGHT_DECAY, OPTIMIZER_NAME, LOSS_NAME, PATIENCE, DEVICE, SAVE_PATH, METRICS_PATH, DROPOUT
)
from src.transforms import get_train_transform, get_val_transform
from src.data_loader import get_data_loaders
from src.models import initialize_model
from src.optimizers import get_optimizer, get_loss_function
from src.training import train_model

def main():
    #  Set up data loaders
    train_loader, val_loader = get_data_loaders(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )
    
    #  Initialize the model
    model = initialize_model(MODEL_NAME, DROPOUT)

    #  Define optimizer and loss function
    optimizer = get_optimizer(model, optimizer_name=OPTIMIZER_NAME, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = get_loss_function(loss_name=LOSS_NAME)

    #  Train the model
    train_model(
        model=model,
        model_name=MODEL_NAME,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=None,  # Add a scheduler if needed
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        device=DEVICE
    )

if __name__ == "__main__":
    main()
