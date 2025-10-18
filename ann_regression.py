import torch
import numpy as np
import streamlit as st
from torch import nn
from torch.optim import Adam
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from sklearn.model_selection import train_test_split
from base_model import BaseModel

class Hyperparameters:
    LEARNING_RATE: float = 0.01
    EPOCHS: int = 1000

    @property
    def learning_rate(self) -> float:
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float):
        if value <= 0:
            self.__learning_rate = self.LEARNING_RATE
        else:
            self.__learning_rate = value

    @property
    def epochs(self) -> int:
        return self.__epochs

    @epochs.setter
    def epochs(self, value: int):
        if value <= 0:
            self.__epochs = self.EPOCHS
        else:
            self.__epochs = value

    def __init__(self, learning_rate: float = LEARNING_RATE, epochs: int = EPOCHS):
        self.__learning_rate = learning_rate
        self.__epochs = epochs

hyper: Hyperparameters = Hyperparameters()

class ANNRegression(BaseModel[nn.Sequential, nn.MSELoss, Adam]):
    PATH_TO_SAVED_MODEL: str = "ann_regression_model.pt"

    @property
    def X(self) -> np.ndarray:
        """Features as a numpy array of shape (N, 2)."""
        return self.__X

    @property
    def y(self) -> np.ndarray:
        """Targets as a numpy array of shape (N,)."""
        return self.__y

    @property
    def X_train(self) -> np.ndarray:
        """Training features as a numpy array of shape (N_train, 2)."""
        if self.__X_train is None:
            raise ValueError("Training data not initialized.")
        return self.__X_train

    @property
    def X_test(self) -> np.ndarray:
        """Testing features as a numpy array of shape (N_test, 2)."""
        if self.__X_test is None:
            raise ValueError("Testing data not initialized.")
        return self.__X_test

    @property
    def y_train(self) -> np.ndarray:
        """Training targets as a numpy array of shape (N_train,)."""
        if self.__y_train is None:
            raise ValueError("Training data not initialized.")
        return self.__y_train

    @property
    def y_test(self) -> np.ndarray:
        """Testing targets as a numpy array of shape (N_test,)."""
        if self.__y_test is None:
            raise ValueError("Testing data not initialized.")
        return self.__y_test

    @property
    def X_train_tensor(self) -> torch.Tensor:
        """Training features as a torch tensor."""
        return torch.from_numpy(self.X_train)

    @property
    def y_train_tensor(self) -> torch.Tensor:
        """Training targets as a torch tensor."""
        return torch.from_numpy(self.y_train).reshape(-1, 1)

    @property
    def X_test_tensor(self) -> torch.Tensor:
        """Testing features as a torch tensor."""
        return torch.from_numpy(self.X_test)

    @property
    def y_test_tensor(self) -> torch.Tensor:
        """Testing targets as a torch tensor."""
        return torch.from_numpy(self.y_test).reshape(-1, 1)

    @property
    def cos_distribution_figure(self) -> Figure:
        """Matplotlib figure showing the distribution of the target values."""
        fig: Figure = plt.figure(figsize=(6, 4))
        ax: Axes = fig.add_subplot(1, 1, 1)
        ax.scatter(range(len(self.y)), self.y, alpha=0.6)
        ax.set_title('Distribution of Target Values')
        ax.set_xlabel('Target Value')
        ax.set_ylabel('Frequency')
        fig.tight_layout()
        return fig
    
    def __init__(self):
        super().__init__()
        N: int = 1000
        BORDER: int = 3
        rng = np.random.default_rng()
        self.__X = rng.uniform(-BORDER, BORDER, size=(N, 2)).astype(np.float32)
        self.__y = (np.cos(2 * self.__X[:, 0]) + np.cos(3 * self.__X[:, 1])).astype(np.float32)
        
        self.__X_train, self.__X_test, self.__y_train, self.__y_test = train_test_split(
            self.__X, self.__y, test_size=0.2
        )

        global hyper
        self.model = self._load_model_architecture()
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=hyper.learning_rate)

    def save(self):
        self.save_model(self.PATH_TO_SAVED_MODEL)

    def load(self):
        self.load_model(self.PATH_TO_SAVED_MODEL)

    def _load_model_architecture(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(2, 128), 
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def train(self, epochs: int = 1000) -> Figure | None:
        train_losses: np.ndarray = np.zeros(epochs)
        for it in range(epochs):
            self.optimizer.zero_grad()

            outputs: torch.Tensor = self.model(self.X_train_tensor)

            loss: torch.Tensor = self.criterion(outputs, self.y_train_tensor)
            loss.backward()
            self.optimizer.step()

            train_losses[it] = loss.item()
        self.save()
        return self._plot_losses(train_losses, "Train Loss")

    def _plot_losses(self, train_losses: np.ndarray, title: str) -> Figure:
        fig, ax = plt.subplots()
        ax.plot(range(len(train_losses)), train_losses, label=title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        return fig

if __name__ == "__main__":
    model: ANNRegression = ANNRegression()
    train_loss_fig: Figure | None = None 
    st.set_page_config(page_title="ANN Regression Example", layout="wide")
    st.title("Artificial Neural Network Regression Example")
    st.write("This example demonstrates a simple ANN regression model.")

    hyperparameters_bar = st.container(border=True)
    with hyperparameters_bar:
        st.header("Hyperparameters")
        learning_rate = st.number_input(
            "Learning Rate", min_value=0.0001, max_value=1.0, value=hyper.learning_rate, step=0.0001, format="%.4f"
        )
        epochs = st.number_input(
            "Epochs", min_value=1, max_value=10000, value=hyper.epochs, step=1
        )
        hyper.learning_rate = learning_rate
        hyper.epochs = epochs
        if st.button("Train Model", disabled=(learning_rate == hyper.LEARNING_RATE and epochs == hyper.EPOCHS)):
            model = ANNRegression()
            train_loss_fig = model.train(epochs=hyper.epochs)
    
    target_distribution, train_loss = st.tabs(["Target Distribution", "Training Loss"])
    with target_distribution:
        st.pyplot(model.cos_distribution_figure)
    with train_loss:
        st.pyplot(model.train(epochs=hyper.epochs))