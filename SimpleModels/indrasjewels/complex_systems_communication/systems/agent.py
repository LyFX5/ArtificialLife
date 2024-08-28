import numpy as np
from typing import List
from copy import copy, deepcopy
from .automaton import FiniteBinaryAutomaton


class Agent(FiniteBinaryAutomaton):
    def __init__(self, state_init: np.ndarray):
        self.__alphabet = [0., 1.]
        assert all([bit in self.__alphabet for bit in state_init]), "state_init is not binary"
        self.__state = state_init
        self.__dimension = self.__state.shape[0] # number of variables / parameters / negentropy (if effective only) / number of bits / information
        self.channels: List[FiniteBinaryAutomaton] = []
    
    @property
    def state(self) -> np.ndarray:
        return self.__state

    def transition(self) -> None:
        # if not communicating with other agent -> free move
        probable_state = np.random.choice(self.__alphabet, self.__dimension).reshape((self.__dimension, 1))
        # if is connected then depends on input
        for channel in self.channels:
            if (probable_state == channel.state).all():
                return
        self.__state = probable_state

    def set_channels(self, channels: List[FiniteBinaryAutomaton]):
        self.channels = channels
        