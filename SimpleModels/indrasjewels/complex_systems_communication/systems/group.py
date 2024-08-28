import numpy as np
from typing import List
from .automaton import FiniteBinaryAutomaton
from .agent import Agent


class Group(FiniteBinaryAutomaton):
    def __init__(self, agents_array: List[Agent], adjacency_matrix: np.ndarray):
        self.agents_array = agents_array
        self.adjacency_matrix = adjacency_matrix

    @property
    def state(self) -> np.ndarray:
        return np.concatenate([agent.state for agent in self.agents_array], axis=1)

    def transition(self) -> None:
        for j in range(len(self.agents_array)):
            self.agents_array[j].transition()

    # def generate_channels_indexes(self, excluded: int):
    #     avaliable_channels_range = len(self.agents_array)
    #     channels_maximal_number = len(self.agents_array) - 1
    #     channels_number = channels_maximal_number # np.random.choice(channels_maximal_number, 1).item()
    #     return list(set(np.random.choice(avaliable_channels_range, 
    #                                      channels_number)) - {excluded})

    #     channels_indexes = self.generate_channels_indexes(j)
    #     channels = [self.agents_array[c] for c in channels_indexes]
        
    def build_net(self):        
        for j in range(len(self.agents_array)):
            self.adjacency_matrix[j][j] = 0
            channels = []
            for c in range(len(self.agents_array)):
                if self.adjacency_matrix[j][c] == 1:
                    channels.append(self.agents_array[c])
            self.agents_array[j].set_channels(channels)
