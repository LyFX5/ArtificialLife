import numpy as np
from .discret_dynamical_system import System
from .system_state_space import ProbabilityDistribution
from .agent import Agent


class Channel(System):
    def __init__(self, agent_source: Agent, agent_reciver: Agent):
        self.agent_source = agent_source
        self.agent_reciver = agent_reciver
        # связь может быть крепче или слабее
        
    @property
    def state(self) -> np.ndarray:
        ...

    def step(self, *args, **kwargs) -> None:
        self.agent_source.output_signal()
        
    def mapping(self, x):
        return 1. * x
        