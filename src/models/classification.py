import torch
import torch.nn as nn


def get_model(num_classes):
    """
    Get the model
    :param num_classes: int
    :return: torch.nn.Module
    """
    model = torch.hub.load(
        "pytorch/vision:v0.10.0", "resnet18", weights="ResNet18_Weights.DEFAULT"
    )
    model.fc = nn.Sequential(nn.Linear(512, num_classes))
    return model
