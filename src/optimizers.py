import torch

def get_optimizer(model, optimizer_name="adam", lr=1e-4, weight_decay=1e-4):
  
    if optimizer_name.lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_loss_function(loss_name="bce"):

    if loss_name.lower() == "bce":
        return torch.nn.BCEWithLogitsLoss()
    elif loss_name.lower() == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    elif loss_name.lower() == "mse":
        return torch.nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
