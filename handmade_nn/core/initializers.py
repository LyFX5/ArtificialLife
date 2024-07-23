import numpy as np
from abc import ABC, abstractmethod


class Initializer(ABC):
    @abstractmethod
    def initialize(prev_neurons: int, cur_neurons: int, next_neurons: int) -> np.ndarray:
        pass


class He(Initializer):
    @staticmethod
    def initialize(prev_neurons: int, cur_neurons: int, next_neurons: int = 0) -> np.ndarray:
        assert prev_neurons * cur_neurons > 0
        return np.random.normal(scale=2 / cur_neurons, size=(prev_neurons, cur_neurons))


class Xavier(Initializer):
    @staticmethod
    def initialize(prev_neurons: int, cur_neurons: int, next_neurons: int) -> np.ndarray:
        assert prev_neurons  * cur_neurons > 0 and next_neurons >= 0
        scale = 2 * np.sqrt(6) / (cur_neurons + next_neurons)
        return (np.random.random(size=(prev_neurons, cur_neurons)) - 0.5) * scale
