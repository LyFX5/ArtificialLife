import numpy as np
from copy import copy, deepcopy
from .automaton import FiniteBinaryAutomaton


class Agent(FiniteBinaryAutomaton):
    def __init__(self, state_init: np.ndarray):
        self.__alphabet = [0., 1.]
        assert all([bit in self.__alphabet for bit in state_init]), "state_init is not binary"
        self.__state = state_init
        self.__dimension = self.__state.shape[0] # number of variables / parameters / negentropy (if effective only) / number of bits / information
    
    @property
    def state(self) -> np.ndarray:
        return self.__state

    def transition(self, input_signal: np.ndarray) -> None:
        # if not communicating with other agent -> free move
        self.__state = np.random.choice(self.__alphabet, self.__dimension).reshape((self.__dimension, 1))
        # if is connected then depends on input
        