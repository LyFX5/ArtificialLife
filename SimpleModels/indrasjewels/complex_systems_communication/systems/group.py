import numpy as np
from typing import List
from .automaton import FiniteBinaryAutomaton
from .agent import Agent


class Group(FiniteBinaryAutomaton):
    def __init__(self, agents_array: List[Agent]):
        self.agents_array = agents_array

    @property
    def state(self) -> np.ndarray:
        return np.concatenate([agent.state for agent in self.agents_array], axis=1)

    def transition(self) -> None:
        for j in range(len(self.agents_array)):
            self.agents_array[j].transition(None)
