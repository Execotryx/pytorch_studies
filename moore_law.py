"""Streamlit app modeling Moore's Law with linear regression in PyTorch.

This module downloads the Moore's Law dataset if needed, transforms the
target to log2 scale, standardizes features/targets, trains a simple linear
model with MSE loss and SGD, and renders loss and fitted results in a
Streamlit UI.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import streamlit as st
import urllib.request as req
from pathlib import Path
from typing import Callable

class MooresLaw:
    """Handle data preparation, training, and visualization for Moore's Law."""
    MOORES_LAW_DATA_URL: str = "https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/tf2.0/moore.csv"
    FILE_NAME: str = "moore.csv"

    @property
    def X(self) -> np.ndarray:
        """Standardized feature array (years) with shape (N, 1)."""
        return self.__X

    @property
    def y(self) -> np.ndarray:
        """Standardized log2 transistor counts with shape (N, 1)."""
        return self.__y

    @property
    def inputs(self) -> torch.Tensor:
        """Tensor view of ``X`` used as model input."""
        return self.__inputs

    @property
    def targets(self) -> torch.Tensor:
        """Tensor view of ``y`` used as regression targets."""
        return self.__targets

    @property
    def model(self) -> nn.Linear:
        """The underlying PyTorch linear model (1 input, 1 output)."""
        return self.__model

    @property
    def criterion(self) -> nn.MSELoss:
        """Mean squared error loss function used for training."""
        return self.__criterion

    @property
    def optimizer(self) -> torch.optim.SGD:
        """SGD optimizer configured with momentum for faster convergence."""
        return self.__optimizer

    def __get_data_if_needed(self) -> None:
        """Download the Moore's Law CSV file if it's not present locally."""
        out: Path = Path(self.FILE_NAME)
        if not out.exists():
            req.urlretrieve(self.MOORES_LAW_DATA_URL, out)

    def __scale(self) -> None:
        """Standardize features and targets to zero mean and unit variance."""
        mx: float = self.X.mean()
        sx: float = self.X.std()

        my: float = self.y.mean()
        sy: float = self.y.std()

        self.__X = (self.X - mx) / sx
        self.__y = (self.y - my) / sy

    def __init__(self):
        """Load data, apply log transform and scaling, and build the model."""
        self.__get_data_if_needed()
        data: np.ndarray = pd.read_csv(self.FILE_NAME, header=None).values

        self.__X: np.ndarray = data[:, 0].reshape(-1, 1)
        self.__y: np.ndarray = data[:, 1].reshape(-1, 1)

        self.__y = np.log2(self.y)
        self.__scale()

        self.__X = self.__X.astype(np.float32)
        self.__y = self.__y.astype(np.float32)

        self.__inputs: torch.Tensor = torch.from_numpy(self.X)
        self.__targets: torch.Tensor = torch.from_numpy(self.y)

        self.__model: nn.Linear = nn.Linear(1, 1)
        self.__criterion: nn.MSELoss = nn.MSELoss()
        self.__optimizer: torch.optim.SGD = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.7)

    def train(self, epochs: int = 100, progress_callback: Callable[[int, float], None] | None = None) -> Figure:
        """
        Train the model for `epochs` steps. If `progress_callback` is provided it will be called
        after every epoch with two arguments: (epoch_number: int, loss: float).

        Returns:
            Figure: Matplotlib figure containing the loss plot.
        """
        losses: list[float] = []
        for it in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)

            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

            if progress_callback is not None:
                try:
                    progress_callback(it + 1, loss.item())
                except Exception:
                    # Don't let progress UI errors stop training
                    pass

        figure, plotted = plt.subplots()
        # epochs on x-axis, loss on y-axis
        plotted.scatter(range(1, len(losses) + 1), losses, label="Training loss", color='blue')
        plotted.set_xlabel("Epoch")
        plotted.set_ylabel("Loss")
        plotted.legend()
        plt.tight_layout()
        return figure

    def show_results(self) -> Figure:
        """Create a matplotlib figure showing data points and fitted line."""
        figure, plotted = plt.subplots()
        plotted.scatter(self.X, self.y, label="Original data", color='blue')
        predicted = self.model(self.inputs).detach().numpy()
        plotted.plot(self.X, predicted, label="Fitted line", color='red')
        plotted.legend()
        return figure

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.title("Moore's Law with Linear Regression")
    st.write("""
        Moore's Law is the observation that the number of transistors in a dense integrated circuit
        doubles about every two years. Here we fit a line to the log base 2 of the number of transistors
        as a function of time.
    """)
    model: MooresLaw = MooresLaw()
    losses_tab, results_tab = st.tabs(["Losses", "Results"])
    epochs = 100

    with losses_tab:
        with st.spinner("Training the model..."):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def _cb(epoch: int, loss_value: float) -> None:
                # update progress and status text
                try:
                    percent = int(epoch / epochs * 100)
                except Exception:
                    percent = 0
                progress_bar.progress(percent)
                status_text.text(f"Epoch {epoch}/{epochs}, Loss: {loss_value:.4f}")

            figure = model.train(epochs=epochs, progress_callback=_cb)

        st.pyplot(figure)
        st.write("Training complete.")

    with results_tab:
        results_figure: Figure = model.show_results()
        st.pyplot(results_figure)