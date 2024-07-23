from copy import copy
import numpy as np
# import hashlib
import pandas as pd
from typing import Iterable, Iterator
from agent import Agent


class System(Iterator):
    def __init__(self, concentrations_on_layers_init: np.ndarray):
        self.__init_grid(concentrations_on_layers_init)
        self.__time = 0

    def __init_grid(self, concentrations_on_layers_init):
        self.__grid = np.zeros((len(concentrations_on_layers_init), concentrations_on_layers_init.max()))
        for i in range(len(concentrations_on_layers_init)):
            for j in range(concentrations_on_layers_init[i]):
                self.__grid[i, j] = 1

    @property
    def number_of_layers(self) -> int:
        return self.__grid.shape[0]

    @property
    def concentrations_on_layers(self) -> np.ndarray:
        return np.array([self.__grid[i, :].sum() for i in range(self.number_of_layers)])

    @property
    def grid(self) -> dict:
        return copy(self.__grid)

    @property
    def time(self) -> int:
        return self.__time

    @property
    def state(self) -> pd.DataFrame:
        concentrations_on_layers = self.concentrations_on_layers
        return pd.DataFrame({f"layer_{layer}": [concentrations_on_layers[layer]] for layer in range(self.number_of_layers)})

    def agents_on_layer(self, layer: int) -> list[Agent]:
        pass

    def step(self):
        # step agents
        # update grid
        self.__time += 1

    def __next__(self, ) -> pd.DataFrame:
        self.step()
        return self.state

    def __iter__(self, ) -> Iterator:
        return self


