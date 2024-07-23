
class Agent:
    def __init__(self, ID, point):
        self.__ID = ID
        self.__point = point # (layer, position)

    def get_id(self):
        return self.__ID

    def get_point(self):
        return self.__point

    def step(self):
        ...

    def __str__(self):
        return f'ID: {self.__ID}, point: {self.__point}'



