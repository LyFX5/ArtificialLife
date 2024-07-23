
import random

class Agent:
    def __init__(self,val):
        self.cal = val

    def get_action(self,env_state):
        '''
        print()
        print("=============================")
        print("State is ")
        print(env_state)
        print("=============================")
        print()
        '''
        return [round(random.uniform(-1,1),3), round(random.uniform(0,1),3), 0]
