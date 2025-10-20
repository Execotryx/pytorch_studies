import torch
import numpy as np
import streamlit as st
import torchvision.transforms as transforms
from torch import dtype, nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from datetime import datetime, timedelta
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from os.path import exists
from os import system, name

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2470, 0.2435, 0.2610)

train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

test_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

train_dataset: CIFAR10 = CIFAR10(root='./data', train=True, download=True, transform=train_tfms)
test_dataset: CIFAR10 = CIFAR10(root='./data', train=False, download=True, transform=test_tfms)

class CNN(nn.Module):
    DEFAULT_BATCH_SIZE: int = 32
    PATH_TO_SAVED_MODEL: str = 'cnn_cifar10_model.pt'
    
    @property
    def _model(self) -> nn.Sequential:
        """The CNN model architecture."""
        return self.__model

    @property
    def _device(self) -> torch.device:
        """Device on which the model is allocated (CPU or GPU)."""
        return self.__device

    @property
    def _criterion(self) -> nn.CrossEntropyLoss:
        """Loss function used for training."""
        return self.__criterion

    @property
    def _optimizer(self) -> SGD:
        """Optimizer used for training."""
        return self.__optimizer

    @property
    def _batch_size(self) -> int:
        """Batch size used for training."""
        return self.__batch_size

    @_batch_size.setter
    def _batch_size(self, batch_size: int) -> None:
        """Set the batch size for training."""
        if batch_size <= 0:
            self.__batch_size = self.DEFAULT_BATCH_SIZE
        else:
            self.__batch_size = batch_size

    @property
    def _num_workers(self) -> int:
        """Number of workers for data loading."""
        return 4 if self._device.type == 'cuda' else 0

    @property
    def _pin_memory(self) -> bool:
        """Whether to pin memory during data loading."""
        return self._device.type == 'cuda'

    @property
    def _persistent_workers(self) -> bool:
        """Whether to use persistent workers for data loading."""
        return self._device.type == 'cuda'

    @property
    def _train_loader(self) -> DataLoader:
        """DataLoader for the training dataset."""
        return self.__train_loader

    @property
    def _test_loader(self) -> DataLoader:
        """DataLoader for the test dataset."""
        return self.__test_loader

    @property
    def accuracy(self) -> float:
        """Current accuracy of the model."""
        if self.__accuracy == 0.0:
            self.__accuracy = self._evaluate_accuracy()
        return self.__accuracy

    def __init__(self, K: int, 
                 kernel_size: int = 3, 
                 stride: int = 1, 
                 dropout_rate: float = 0.2,
                 batch_size: int = 1) -> None:
        """
        Initialize the CNN architecture.
        Args:
            K (int): Number of filters in the convolutional layers.
            input_channels (int): Number of input channels. Default is 3 (RGB images).
            kernel_size (int): Size of the convolutional kernels. Default is 3.
            stride (int): Stride for the convolutional layers. Default is 2.
        """
        super(CNN, self).__init__()
        self.__batch_size: int = batch_size
        self.__accuracy: float = 0.0
        self._setup_training(kernel_size, stride, dropout_rate, K)
        if exists(self.PATH_TO_SAVED_MODEL):
            self.load()

    def _evaluate_accuracy(self) -> float:
        """Evaluate the accuracy of the model on the test dataset."""
        self._model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for inputs, targets in self._test_loader:
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                outputs = self._model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        accuracy = 100 * (correct / total)
        print(f'Accuracy of the model on the test images: {accuracy:.2f}%')
        return accuracy

    def _setup_training(self, kernel_size: int, stride: int, dropout_rate: float, K: int) -> None:
        """
        Setup the training components including model, optimizer, loss function, and data loaders.
         Args:
            kernel_size (int): Size of the convolutional kernels.
            stride (int): Stride for the convolutional layers.
            dropout_rate (float): Dropout rate for regularization.
            K (int): Number of output classes.
        """

        self.__device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.__model: nn.Sequential = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.LazyLinear(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, K)
        )

        self._model.to(self._device)

        self.__criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self.__optimizer: SGD = SGD(self._model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)

        self.__train_loader: DataLoader = DataLoader(
            dataset=train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers
        )

        self.__test_loader: DataLoader = DataLoader(
            dataset=test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        out: torch.Tensor = self._model(x)
        return out

    def save(self, path: str = PATH_TO_SAVED_MODEL) -> None:
        """Save the model state to a file."""
        torch.save(self._model.state_dict(), path)

    def load(self, path: str = PATH_TO_SAVED_MODEL) -> None:
        """Load the model state from a file."""
        self._model.load_state_dict(torch.load(path, map_location=self._device))

    def _amp_setup(self) -> tuple[autocast, GradScaler | None]:
        if self._device.type == 'cuda':
            use_bf16: bool = torch.cuda.is_bf16_supported()
            amp_dtype: dtype = torch.bfloat16 if use_bf16 else torch.float16
            autocast_context = autocast(device_type='cuda', dtype=amp_dtype)
            scaler: GradScaler | None = None if use_bf16 else GradScaler(device=self._device.type)
            return autocast_context, scaler
        raise RuntimeError("AMP is only supported on CUDA devices.")

    def train_cnn(self, epochs: int = 15) -> Figure:
        torch.cuda.memory.empty_cache()
        try:
            self._model.train()
            train_losses = np.zeros(epochs)
            test_losses = np.zeros(epochs)

            autocast_context, scaler = self._amp_setup()

            for it in range(epochs):
                t0: datetime = datetime.now()
                batch_loss_sum = 0.0
                batch_count = 0

                for inputs, targets in self._train_loader:
                    inputs = inputs.to(self._device, non_blocking=True)
                    targets = targets.to(self._device, non_blocking=True)

                    self._optimizer.zero_grad(set_to_none=True)

                    with autocast_context:
                        outputs: torch.Tensor = self._model(inputs)
                        loss: torch.Tensor = self._criterion(outputs, targets)

                    if scaler is None:
                        loss.backward()
                        self._optimizer.step()
                    else:
                        scaler.scale(loss).backward()
                        scaler.step(self._optimizer)
                        scaler.update()

                    batch_loss_sum += loss.item()
                    batch_count += 1

                train_losses[it] = batch_loss_sum / max(1, batch_count)

                # --- evaluation pass (no grad, eval mode) ---
                self._model.eval()
                with torch.inference_mode():
                    test_loss_sum = 0.0
                    test_batches = 0
                    for inputs, targets in self._test_loader:
                        inputs = inputs.to(self._device, non_blocking=True)
                        targets = targets.to(self._device, non_blocking=True)
                        outputs = self._model(inputs)
                        loss = self._criterion(outputs, targets)
                        test_loss_sum += float(loss)
                        test_batches += 1
                    test_losses[it] = test_loss_sum / max(1, test_batches)
                    dt: timedelta = datetime.now() - t0
                    print(f"Epoch {it+1}/{epochs} - Train Loss: {train_losses[it]:.4f}, Test Loss: {test_losses[it]:.4f}, Time: {dt.total_seconds():.2f}s")
                self._model.train()

            fig, ax = plt.subplots()
            ax.plot(range(epochs), train_losses, label="Train Loss")
            ax.plot(range(epochs), test_losses, label="Test Loss")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
            return fig
        except KeyboardInterrupt:
            import gc
            gc.collect()
            torch.cuda.memory.empty_cache()
            # Return an empty figure when training is interrupted
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'Training interrupted', ha='center', va='center', transform=ax.transAxes)
            return fig


if __name__ == "__main__":
    st.set_page_config(page_title="CNN CIFAR-10 Training", layout="wide")
    st.title("CNN CIFAR-10 Classifier Training")
    st.write("This application trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset using PyTorch.")
    st.sidebar.header("Settings")
    epochs = st.sidebar.slider("Epochs", 1, 100, 15)
    batch_size = st.sidebar.slider("Batch Size", 1, 128, 64)
    train_model_button = st.sidebar.button("Train Model")
    cnn_model: CNN = CNN(K=10, batch_size=batch_size)
    training_loss_fig: Figure | None = None
    if train_model_button:
        with st.spinner("Training the model..."):
            system("cls" if name == "nt" else "clear")
            training_loss_fig = cnn_model.train_cnn(epochs=epochs)
        cnn_model.save()
    training, accuracy = st.tabs(["Training", "Accuracy"])
    with training:
        if training_loss_fig:
            st.pyplot(training_loss_fig)
    with accuracy:
        st.write(f"Model Accuracy: {cnn_model.accuracy:.2f}%")