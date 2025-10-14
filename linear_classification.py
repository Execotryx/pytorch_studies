"""Streamlit app for binary linear classification using PyTorch.

This module loads the Breast Cancer Wisconsin dataset, standardizes
the features, trains a simple logistic regression (linear layer + sigmoid)
model with binary cross-entropy loss, and visualizes training/test loss
curves and accuracy inside a Streamlit UI.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
from typing import cast
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from os.path import exists
from base_model import BaseModel

class LinearClassification(BaseModel[nn.Module, nn.Module, optim.Optimizer]):
    """Binary linear classification on the Breast Cancer dataset.

    This class encapsulates data loading, preprocessing with
    ``StandardScaler``, model definition (a linear layer followed by
    a sigmoid), training with ``BCELoss`` and ``Adam`` optimizer, and
    simple evaluation utilities (loss plot and accuracy).
    """

    PATH_TO_SAVED_MODEL: str = "linear_classification_model.pt"

    @property
    def N(self) -> int:
        """Number of training samples (rows) in the dataset."""
        return self.__N

    @property
    def D(self) -> int:   
        """Number of features (columns) after preprocessing."""
        return self.__D

    @property
    def _scaler(self) -> StandardScaler:
        """Fitted ``StandardScaler`` used to standardize features."""
        return self.__scaler

    # region: Properties for raw data as numpy arrays
    @property
    def X_train(self) -> torch.Tensor:
        """Training features as a float32 ``torch.Tensor`` of shape (N, D)."""
        return self.__X_train_tensor

    @property
    def y_train(self) -> torch.Tensor:
        """Training labels as a float32 ``torch.Tensor`` of shape (N, 1)."""
        return self.__y_train_tensor

    @property
    def X_test(self) -> torch.Tensor:
        """Test features as a float32 ``torch.Tensor`` of shape (N_test, D)."""
        return self.__X_test_tensor

    @property
    def y_test(self) -> torch.Tensor:
        """Test labels as a float32 ``torch.Tensor`` of shape (N_test, 1)."""
        return self.__y_test_tensor
    # endregion

    def _initialize_from_scratch(self) -> None:
        """Load and preprocess data, build model, and setup tensors from scratch."""
        # force sklearn to return X, y tuple for static typing
        X, y = cast(tuple[np.ndarray, np.ndarray], load_breast_cancer(return_X_y=True))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.__N = X_train.shape[0]
        self.__D = X_train.shape[1]
        self.__scaler = StandardScaler()
        X_train_scaled = self.__scaler.fit_transform(X_train)
        X_test_scaled = self.__scaler.transform(X_test)
        # build model architecture
        self._model = self._load_model_architecture()
        # prepare tensors
        self.__X_train_tensor = torch.from_numpy(X_train_scaled.astype("float32"))
        self.__y_train_tensor = torch.from_numpy(y_train.astype("float32")).reshape(-1, 1)
        self.__X_test_tensor = torch.from_numpy(X_test_scaled.astype("float32"))
        self.__y_test_tensor = torch.from_numpy(y_test.astype("float32")).reshape(-1, 1)

    def _setup_training(self) -> None:
        """Setup loss function and optimizer if the model is available."""
        try:
            model = self._model
        except AttributeError:
            return
        self._criterion = nn.BCELoss()
        self._optimizer = optim.Adam(model.parameters())

    def _load_model_architecture(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(30, 1),
            nn.Sigmoid()
        )

    def __init__(self) -> None:
        """Initialize data, preprocessing, model, loss, and optimizer."""
        super().__init__()
        if not exists(self.PATH_TO_SAVED_MODEL):
            self._initialize_from_scratch()
        else:
            self.load_model(self.PATH_TO_SAVED_MODEL)
        self._setup_training()

    def train(self, epochs: int = 100) -> Figure | None:
        """
        Train the model for `epochs` steps.

        Returns:
            Figure: Matplotlib figure containing the loss plot.
        """
        try:
            model = self._model
            criterion = self._criterion
            optimizer = self._optimizer
        except AttributeError:
            return None

        train_losses: np.ndarray = np.zeros(epochs)
        test_losses: np.ndarray = np.zeros(epochs)
        for it in range(epochs):
            optimizer.zero_grad()
            outputs: torch.Tensor = model(self.X_train)
            loss: torch.Tensor = criterion(outputs, self.y_train)

            loss.backward()
            optimizer.step()

            outputs_test: torch.Tensor = model(self.X_test)
            loss_test: torch.Tensor = criterion(outputs_test, self.y_test)

            train_losses[it] = loss.item()
            test_losses[it] = loss_test.item()

        figure, plotted = plt.subplots()
        figure = cast(Figure, figure)
        plotted = cast(Axes, plotted)
        # epochs on x-axis, loss on y-axis
        plotted.plot(train_losses, label="Training loss", color='blue')
        plotted.plot(test_losses, label="Test loss", color='orange')
        plotted.set_xlabel("Epoch")
        plotted.set_ylabel("Loss")
        plotted.legend()
        plt.tight_layout()
        return figure

    def save(self):
        """Save the trained model to a file."""
        self.save_model(self.PATH_TO_SAVED_MODEL)

    def load(self):
        """Load a trained model from a file."""
        self.load_model(self.PATH_TO_SAVED_MODEL)

    @property
    def accuracy(self) -> str:
        """Compute formatted train/test accuracy string.

        Returns:
            str: A human-readable accuracy summary for train and test sets.
                 Returns messages if data is empty or model is unavailable.
        """
        if self.X_train.shape[0] == 0 or self.X_test.shape[0] == 0:
            return "Train or test set is empty"

        try:
            model = self._model
        except AttributeError:
            return "Model is not trained"
        
        with torch.no_grad():
            p_train_t: torch.Tensor = model(self.X_train)
            p_train: np.ndarray = np.round(p_train_t.numpy())
            train_acc: float = float(np.mean(p_train == self.y_train.numpy()))

            p_test_t: torch.Tensor = model(self.X_test)
            p_test: np.ndarray = np.round(p_test_t.numpy())
            test_acc: float = float(np.mean(p_test == self.y_test.numpy()))
            return f"Train accuracy: {train_acc * 100:.2f}%, Test accuracy: {test_acc * 100:.2f}%"


if __name__ == "__main__":
    st.set_page_config(page_title="Linear Classification", layout="wide")
    model = LinearClassification()
    st.title("Linear Classification")
    st.write("Using the Breast Cancer Wisconsin (Diagnostic) Dataset from sklearn.datasets")
    losses, accuracy = st.tabs(["Losses", "Accuracy"])
    with losses:
        st.write("## Losses")
        st.pyplot(model.train())
    with accuracy:
        st.write("## Accuracy")
        st.write(model.accuracy)
        model.save()