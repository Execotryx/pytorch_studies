import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from base_model import BaseModel

training_dataset = MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_dataset = MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ERU(nn.Module):
    """
    Exponential Root Unit (ERU) Activation Function. Go here for details: https://arxiv.org/pdf/1804.11237
    """
    def __init__(self, r: float = 2.0):
        super().__init__()
        self.r = float(r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r: float = self.r
        pos: torch.Tensor = torch.pow(r*r*x + 1.0, 1.0/r) - 1.0/r
        neg: torch.Tensor = torch.exp(r*x) - 1.0/r
        return torch.where(x >= 0, pos, neg)

class ORU(nn.Module):
    """
    Odd Root Unit (ORU) Activation Function. Go here for details: https://arxiv.org/pdf/1804.11237
    """
    def __init__(self, r: float = 2.0):
        super().__init__()
        self.r = float(r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r: float = self.r
        return torch.sign(x) * (torch.pow(r*r*torch.abs(x) + 1.0, 1.0/r) - 1.0)

class MNIST_Model(BaseModel[nn.Sequential, nn.CrossEntropyLoss, torch.optim.Adam]):
    INPUT_FEATURES: int = 784  # 28*28
    OUTPUT_CLASSES: int = 10   # digits 0-9
    PATH_TO_SAVED_MODEL: str = "mnist_model.pt"
    
    def __init__(self):
        super().__init__()

        self._model = nn.Sequential(
            nn.Linear(self.INPUT_FEATURES, 128),
            ERU(r=3.5),
            nn.Linear(128, self.OUTPUT_CLASSES)
        )

        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        self._model.to(device)