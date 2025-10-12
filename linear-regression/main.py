"""Streamlit app demonstrating simple linear regression with PyTorch.

This module generates synthetic data, fits a linear model using MSE loss
and SGD, logs loss per epoch, and visualizes the fitted line alongside
the original points in a Streamlit UI.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import streamlit as st

class LinearRegressionModel:
    """Encapsulates synthetic data generation and linear regression training."""
    N: int = 20

    @property
    def X(self) -> np.ndarray:
        """Feature array of shape (N, 1) used for training/plotting."""
        return self.__X

    @property
    def Y(self) -> np.ndarray:
        """Target array of shape (N, 1) with noise added."""
        return self.__Y

    @property
    def model(self) -> nn.Linear:
        """The underlying PyTorch linear model (1 input, 1 output)."""
        return self.__model

    @property
    def criterion(self) -> nn.MSELoss:
        """Mean squared error loss used during training."""
        return self.__criterion

    @property
    def inputs(self) -> torch.Tensor:
        """Tensor view of ``X`` used as model input."""
        return self.__inputs

    @property
    def targets(self) -> torch.Tensor:
        """Tensor view of ``Y`` used as regression targets."""
        return self.__targets

    @property
    def optimizer(self) -> torch.optim.SGD:
        """Stochastic gradient descent optimizer used to fit the model."""
        return self.__optimizer

    def __init__(self):
        """Generate synthetic data and prepare model, loss, optimizer, tensors."""
        self.__X: np.ndarray = np.random.random(self.N) * 10 - 5
        self.__Y: np.ndarray = 0.5 * self.__X - 1 + np.random.randn(self.N)

        self.__model: nn.Linear = nn.Linear(1, 1)
        self.__criterion: nn.MSELoss = nn.MSELoss()
        self.__optimizer: torch.optim.SGD = torch.optim.SGD(self.model.parameters(), lr=0.01)

        self.__X = self.__X.reshape(self.N, 1)
        self.__Y = self.__Y.reshape(self.N, 1)

        self.__inputs: torch.Tensor = torch.from_numpy(self.__X.astype(np.float32))
        self.__targets: torch.Tensor = torch.from_numpy(self.__Y.astype(np.float32))

    def train(self, epochs: int = 30):
        """Train the model yielding a status message on each epoch.

        Args:
            epochs: Total number of optimization steps to run.

        Yields:
            str: A formatted message with current epoch and loss.
        """
        losses = []
        for it in range(epochs):
            self.optimizer.zero_grad()
            outputs: torch.Tensor = self.model(self.inputs)
            loss: torch.Tensor = self.criterion(outputs, self.targets)

            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

            yield f"Epoch {it+1}/{epochs}, Loss: {loss.item():.4f}"

    def show_results(self) -> Figure:
        """Create a matplotlib figure showing data points and fitted line."""
        figure, plotted = plt.subplots()
        plotted.scatter(self.X, self.Y, label="Original data", color='blue')
        predicted = self.model(self.inputs).detach().numpy()
        plotted.plot(self.X, predicted, label="Fitted line", color='red')
        plotted.legend()
        return figure

st.set_page_config(layout="wide")
st.title("Linear Regression with PyTorch")
model = LinearRegressionModel()
losses, results = st.tabs(["Losses", "Results"])
with losses:
    st.spinner("Training the model...")
    for message in model.train(epochs=100):
        st.write(message)
    st.write("Training complete.")
with results:
    figure = model.show_results()
    st.pyplot(figure)