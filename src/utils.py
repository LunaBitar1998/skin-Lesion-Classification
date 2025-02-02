def early_stopping(val_metric, best_metric, counter, patience, mode="max", min_delta=0.001):
    """
    Early stopping utility to monitor training performance.
    
    Args:
        val_metric (float): Current value of the monitored metric (e.g., validation accuracy).
        best_metric (float): Best value of the monitored metric so far.
        counter (int): Counter for epochs without improvement.
        patience (int): Number of epochs to wait before stopping.
        mode (str): "max" for accuracy, "min" for loss.
        min_delta (float): Minimum improvement required to reset the counter.

    Returns:
        tuple: (updated_best_metric, updated_counter, stop_training)
    """
    if mode == "max":
        if val_metric > best_metric + min_delta:
            best_metric = val_metric
            counter = 0
        else:
            counter += 1
    elif mode == "min":
        if val_metric < best_metric - min_delta:
            best_metric = val_metric
            counter = 0
        else:
            counter += 1

    stop_training = counter >= patience
    return best_metric, counter, stop_training

import matplotlib.pyplot as plt

def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_path):
    
    plt.figure(figsize=(14, 6))

    # Plot Losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Plot Accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
