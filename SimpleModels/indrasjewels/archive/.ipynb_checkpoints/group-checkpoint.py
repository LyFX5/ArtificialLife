import numpy as np
from typing import List
from .automaton import FiniteBinaryAutomaton
from .agent import Agent


class Group(FiniteBinaryAutomaton):
    def __init__(self, agents_array: List[Agent], state_init: np.ndarray):
        super(Group, self).__init__(state_init=state_init)
        self.agents_array = agents_array
        self.build_net()
        
    def build_net(self):
        for s in self.agents_array:
            ...
            
        
    @property
    def state(self) -> np.ndarray:
        return np.array([agent.state for agent in self.agents_array])
    
    def step(self):
        ...
        
    def information(self):
        # что есть групповая информация?
        # каждого агента характеризует некоторая доля индивидуальности
        
        return 
        