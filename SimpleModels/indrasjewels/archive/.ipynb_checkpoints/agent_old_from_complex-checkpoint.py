

class AgentLinear(Agent):
    def __init__(self, init_state: np.ndarray, distribution: ProbabilityDistribution, input_signal_dimension=1, a1=0.1, a2=1):
        self.__state = init_state
        self.distribution = distribution
        self.state_dimension = np.shape(init_state)[0]
        self.input_signal_dimension = input_signal_dimension
        self.output_signal_dimension = self.input_signal_dimension
        self.a1 = a1
        self.a2 = a2
        self.init_linear_dynamics()
        
    @property
    def state(self) -> np.ndarray:
        return self.__state
        
    @property
    def output_signal(self) -> np.ndarray:
        # np.matmul(self.C, self.__state) + self.distribution.choice[0]
        # output_signal = 0
        # output_signal = self.state.sum()
        output_signal = self.state[0]
        #output_signal += self.distribution.choice[0]
        output_signal = output_signal.reshape((self.output_signal_dimension, 1))
        return output_signal

    def step(self, input_signal: np.ndarray) -> None:
        # x_next = A * x + B * u + e
        self.__state = np.matmul(self.A, self.state)
        self.__state += np.matmul(self.B, input_signal)
        # self.__state += self.distribution.choice
        
    def init_linear_dynamics(self):
        self.A = np.eye(self.state_dimension) # inner dynamics / free move
        self.A[0][0] = -self.a1
        self.A[0][1] = 1
        self.A[1][0] = -self.a2
        self.A[1][1] = 0
        self.B = np.ones((self.state_dimension, self.input_signal_dimension)) # input mapping
        self.C = np.zeros((self.output_signal_dimension, self.state_dimension)) # output mapping
        self.C[0] = 1
        
        
