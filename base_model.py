import torch
from torch.nn import Module
from os.path import exists
from abc import ABC, abstractmethod
from matplotlib.figure import Figure
from matplotlib.axes import Axes

class BaseModel(ABC):
    DEFAULT_EPOCHS: int = 100

    @property
    def _model(self) -> Module | None:
        """The underlying PyTorch model."""
        return self.__model

    @property
    def _criterion(self) -> Module | None:
        """The loss function used for training."""
        return self.__criterion

    @_criterion.setter
    def _criterion(self, criterion: Module) -> None:
        self.__criterion = criterion

    @property
    def _optimizer(self) -> torch.optim.Optimizer | None:
        """The optimizer used for training."""
        return self.__optimizer

    @_optimizer.setter
    def _optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        self.__optimizer = optimizer
    
    def __init__(self):
        self.__model: Module | None = None
        self.__criterion: Module | None = None
        self.__optimizer: torch.optim.Optimizer | None = None

    @abstractmethod
    def train(self, epochs: int = DEFAULT_EPOCHS) -> Figure | None:
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass

    def save_model(self, filepath: str):
        if self.__model is not None:
            torch.save(self.__model.state_dict(), filepath)

    def load_model(self, filepath: str):
        if exists(filepath):
            self.__model = self.__load_model_architecture()
            self.__model.load_state_dict(torch.load(filepath))

    @abstractmethod
    def __load_model_architecture(self) -> Module:
        pass
