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

    @property
    def _train_loader(self) -> torch.utils.data.DataLoader:
        """DataLoader for training dataset."""
        return self.__train_loader

    @property
    def _test_loader(self) -> torch.utils.data.DataLoader:
        """DataLoader for test dataset."""
        return self.__test_loader

    @property
    def _device(self) -> torch.device:
        """Device (CPU or GPU) used for training and inference."""
        return self.__device

    @property
    def _radix(self) -> float:
        """Radix parameter for the ERU activation function."""
        return self.__radix

    @property
    def _accuracy(self) -> float:
        if self.__accuracy == 0.0:
            self.__accuracy = self._evaluate_accuracy()
        return self.__accuracy

    def _evaluate_accuracy(self) -> float:
        n_correct: int = 0
        n_total: int = 0

        for inputs, targets in self._test_loader:
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            inputs = inputs.reshape(-1, self.INPUT_FEATURES)

            outputs = self._model(inputs)
            _, predicted = torch.max(outputs, 1)
            n_total += targets.size(0)
            n_correct += (predicted == targets).sum().item()

        return n_correct / n_total if n_total > 0 else 0.0

    def _load_model_architecture(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.INPUT_FEATURES, 128),
            ERU(r=self._radix),
            nn.Linear(128, self.OUTPUT_CLASSES)
        )
    
    def __init__(self, batch_size: int = 128, radix: float = 3.5):
        super().__init__()
        self.__accuracy: float = 0.0

        self.__radix = radix

        self._model = self._load_model_architecture()

        self.__device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self._device}")
        self._model.to(self._device)

        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.Adam(self._model.parameters())

        self.__train_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            dataset=training_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        self.__test_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

    def save(self):
        self.save_model(self.PATH_TO_SAVED_MODEL)

    def load(self):
        self.load_model(self.PATH_TO_SAVED_MODEL)

    def train(self, epochs: int = 10) -> Figure | None:
        train_losses = np.zeros(epochs)
        test_losses = np.zeros(epochs)

        for it in range(epochs):
            train_loss = []
            for inputs, targets in self._train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                inputs = inputs.reshape(-1, self.INPUT_FEATURES)
                self._optimizer.zero_grad()

                outputs = self._model(inputs)
                loss = self._criterion(outputs, targets)

                loss.backward()
                self._optimizer.step()
                train_loss.append(loss.item())
            train_losses[it] = np.mean(train_loss)

            test_losses = []
            for inputs, targets in self._test_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                inputs = inputs.reshape(-1, self.INPUT_FEATURES)

                outputs = self._model(inputs)
                loss = self._criterion(outputs, targets)
                test_losses.append(loss.item())
            test_losses[it] = np.mean(test_losses)

        figure, plotted = plt.subplots()
        plotted.plot(range(epochs), train_losses, label="Train Loss")
        plotted.plot(range(epochs), test_losses, label="Test Loss")
        plotted.set_xlabel("Epoch")
        plotted.set_ylabel("Loss")
        plotted.legend()
        return figure
        