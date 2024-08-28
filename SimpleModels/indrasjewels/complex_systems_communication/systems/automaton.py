from abc import ABC, abstractmethod
import numpy as np


class FiniteBinaryAutomaton(ABC):
    
    @property
    @abstractmethod
    def state(self) -> np.ndarray:
        return self.__state

    @abstractmethod
    def transition(self, *args, **kwargs) -> None:
        ...
