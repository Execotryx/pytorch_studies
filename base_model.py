import torch
from torch.nn import Module
from os.path import exists
from abc import ABC, abstractmethod
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Generic, TypeVar

M = TypeVar('M', bound=Module)
C = TypeVar('C', bound=Module)
O = TypeVar('O', bound=torch.optim.Optimizer)

class BaseModel(Generic[M, C, O], ABC):
    DEFAULT_EPOCHS: int = 100

    @property
    def _model(self) -> M:
        """The underlying PyTorch model."""
        if self.__model is None:
            raise AttributeError("Model has not been set.")
        return self.__model

    @_model.setter
    def _model(self, model: M) -> None:
        self.__model = model

    @property
    def _criterion(self) -> C:
        """The loss function used for training."""
        if self.__criterion is None:
            raise AttributeError("Criterion has not been set.")
        return self.__criterion

    @_criterion.setter
    def _criterion(self, criterion: C) -> None:
        self.__criterion = criterion

    @property
    def _optimizer(self) -> O:
        """The optimizer used for training."""
        if self.__optimizer is None:
            raise AttributeError("Optimizer has not been set.")
        return self.__optimizer

    @_optimizer.setter
    def _optimizer(self, optimizer: O) -> None:
        self.__optimizer = optimizer
    
    def __init__(self):
        # initialize private attributes
        self.__model: M | None = None
        self.__criterion: C | None = None
        self.__optimizer: O | None = None

    @abstractmethod
    def train(self, epochs: int = DEFAULT_EPOCHS) -> Figure | None:
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    def save_model(self, filepath: str) -> None:
        """Save model state to disk if model exists."""
        if self.__model is not None:
            torch.save(self.__model.state_dict(), filepath)

    def load_model(self, filepath: str) -> None:
        """Load model state from disk if file exists."""
        if exists(filepath):
            # reconstruct architecture then load state
            self.__model = self._load_model_architecture()
            state_dict = torch.load(filepath)
            self.__model.load_state_dict(state_dict)

    @abstractmethod
    def _load_model_architecture(self) -> M:
        pass
