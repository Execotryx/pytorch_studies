import torch
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torchvision import transforms
from torchvision.datasets import MNIST
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from base_model import BaseModel
import streamlit as st

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
    def _accuracy(self) -> float:
        if self.__accuracy == 0.0:
            self.__accuracy = self._evaluate_accuracy()
        return self.__accuracy

    def _evaluate_accuracy(self) -> float:
        n_correct: int = 0
        n_total: int = 0
        self._model.eval()
        with torch.inference_mode():  # no grad + slightly leaner than no_grad
            for inputs, targets in self._test_loader:
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                inputs = inputs.reshape(-1, self.INPUT_FEATURES)
                outputs = self._model(inputs)
                _, predicted = torch.max(outputs, 1)
                n_total += targets.size(0)
                n_correct += (predicted == targets).sum().item()
        return n_correct / n_total if n_total > 0 else 0.0


    def _load_model_architecture(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self.INPUT_FEATURES, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, self.OUTPUT_CLASSES)
        )
    
    def __init__(self, batch_size: int = 128):
        super().__init__()
        self.__accuracy: float = 0.0
        
        self._model = self._load_model_architecture()
        self._optimizer = torch.optim.Adam(self._model.parameters())
        self._criterion = nn.CrossEntropyLoss()

        self.__device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self._device}")
        self._model.to(self._device)

        # DataLoaders: pin + workers when on CUDA
        num_workers = 4 if self._device.type == "cuda" else 0
        pin = self._device.type == "cuda"
        self.__train_loader = torch.utils.data.DataLoader(
            dataset=training_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=(num_workers > 0)
        )
        self.__test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=(num_workers > 0)
        )

        # Optional global toggles (speed with no extra memory cost)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # good when shapes are stable


    def save(self):
        self.save_model(self.PATH_TO_SAVED_MODEL)

    def load(self):
        self.load_model(self.PATH_TO_SAVED_MODEL)

    def train(self, epochs: int = 10) -> Figure | None:
        self._model.train()
        train_losses = np.zeros(epochs)
        test_losses = np.zeros(epochs)

        scaler = GradScaler()

        for it in range(epochs):
            batch_loss_sum = 0.0
            batch_count = 0

            for inputs, targets in self._train_loader:
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)
                inputs = inputs.reshape(-1, self.INPUT_FEATURES)

                self._optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=(self._device.type=="cuda")):
                    outputs = self._model(inputs)
                    loss = self._criterion(outputs, targets)

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
                    inputs = inputs.reshape(-1, self.INPUT_FEATURES)
                    outputs = self._model(inputs)
                    loss = self._criterion(outputs, targets)
                    test_loss_sum += float(loss)
                    test_batches += 1
                test_losses[it] = test_loss_sum / max(1, test_batches)
            self._model.train()

        fig, ax = plt.subplots()
        ax.plot(range(epochs), train_losses, label="Train Loss")
        ax.plot(range(epochs), test_losses, label="Test Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
        return fig


if __name__ == "__main__":
    st.set_page_config(page_title="MNIST Digit Classifier", layout="wide")
    model = MNIST_Model()
    st.title("MNIST Digit Classifier")
    st.write("This is a simple feedforward neural network trained on the MNIST dataset.")
    
    if st.button("Train Model"):
        with st.spinner("Training the model..."):
            fig = model.train(epochs=10)
            model.save()
            st.success("Model trained and saved!")
            if fig:
                st.pyplot(fig)
            st.write(f"Model Accuracy: {model._accuracy * 100:.2f}%")
    
    if st.button("Load Model"):
        with st.spinner("Loading the model..."):
            model.load()
            st.success("Model loaded!")
            st.write(f"Model Accuracy: {model._accuracy * 100:.2f}%")