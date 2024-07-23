from enum import Enum
import numpy as np

# computation constants
ACCURACY = 3

# system constants
SPACE_DIMENSION = 5
AGENTS_NUMBER = 10
MESSAGE_VELOCITY = 1

# agent constants
DNA_SIZE = 10
GENE_VALUES_NUMBER = 5


class GeneFeatureMap(Enum):
    INITIAL_NEIGHBORS_NUMBER = 0
    INITIAL_SOCIABILITY = 1
    INITIAL_ENERGY_CAPACITY = 2
    INITIAL_ADDRESS_FIRST_DIMENSION = 3
    INITIAL_ADDRESS_LAST_DIMENSION = INITIAL_ADDRESS_FIRST_DIMENSION.value + SPACE_DIMENSION - 1
    INITIAL_DISTANCE_TO_NEIGHBORS = INITIAL_ADDRESS_FIRST_DIMENSION.value + SPACE_DIMENSION
    n = INITIAL_ADDRESS_FIRST_DIMENSION.value + SPACE_DIMENSION + 1


class Agent:
    def __init__(self):
        # что то будет влиять на начальное днк
        self.dna = np.random.randint(0, GENE_VALUES_NUMBER, size=DNA_SIZE)
        print(f"{self.dna=}")

        self.address = self.dna[
                                    GeneFeatureMap.INITIAL_ADDRESS_FIRST_DIMENSION.value :
                                    GeneFeatureMap.INITIAL_ADDRESS_LAST_DIMENSION.value+1
                                    ]
        print(f"{self.address=}")

        self.neighbors_addresses = []

        # self.neighbors_addresses = [self.address +
        #                             self.dna[GeneFeatureMap.INITIAL_DISTANCE_TO_NEIGHBORS.value] * np.random.rand(SPACE_DIMENSION)
        #                             for i in range(self.dna[GeneFeatureMap.INITIAL_NEIGHBORS_NUMBER.value])]
        # print(f"{self.neighbors_addresses=}")

    def send_message(self): # кому, что. то какое сообщение он отправит зависит от его сущности, которая меняется под воздействием внешних сообщений
        pass

    def receive_message(self): # принятое сообщение как то влияет меняет агента (его сущность меняется, он меняется)
        pass







