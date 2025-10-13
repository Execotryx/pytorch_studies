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

    @_model.setter
    def _model(self, model: Module) -> None:
        self.__model = model

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
        self.__model = None  # type: Module | None
        self.__criterion = None  # type: Module | None
        self.__optimizer = None  # type: torch.optim.Optimizer | None

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
        if self._model:
            torch.save(self._model.state_dict(), filepath)

    def load_model(self, filepath: str) -> None:
        """Load model state from disk if file exists."""
        if exists(filepath):
            # reconstruct architecture then load state
            self.__model = self._load_model_architecture()
            state_dict = torch.load(filepath)
            self.__model.load_state_dict(state_dict)

    @abstractmethod
    def _load_model_architecture(self) -> Module:
        pass
