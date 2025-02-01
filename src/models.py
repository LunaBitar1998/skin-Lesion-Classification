import torch
import torch.nn as nn
from torchvision import models

def initialize_model(model_name, dropout=0.5, num_classes=1):
    if model_name == "resnet101":
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.fc.in_features, num_classes)
        )

    elif model_name == "densenet_121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.classifier.in_features, num_classes)
        )

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )

    elif model_name == "efficientnet_b4":
        model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )

    elif model_name == "efficientnet_b7":
        model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.IMAGENET1K_V1)
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )

    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(model.classifier[-1].in_features, num_classes)
        )

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model
