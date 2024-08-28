import numpy as np
from scipy.stats import multivariate_normal
    
    
class ProbabilityDistribution: # over state space
    def __init__(self, number_of_states, dimension, mean, cov):
        self.number_of_states = number_of_states
        self.dimension = dimension
        self.mean = mean
        self.cov = cov
        self.states = np.linspace(0, self.number_of_states, self.number_of_states, endpoint=False)
        self.probabilities = multivariate_normal.pdf(self.states, mean=self.mean, cov=self.cov)
        self.probabilities = self.softmax(self.probabilities)
        
    def softmax(self, a: np.ndarray) -> np.ndarray:
        return np.exp(a) / np.exp(a).sum()        
        
    def choice(self) -> np.ndarray:
        return np.random.choice(self.number_of_states, 
                                self.dimension, 
                                p=self.probabilities).reshape(self.dimension, 1)
    
    def information(self, state):
        assert state in self.states, 'the state does not exist'
        return -np.log2(self.probabilities[state])
    
    def entropy(self):
        return np.dot(self.probabilities, -np.log2(self.probabilities))
    
    def set_entropy(self, entropy: float):
        # 1/2 * np.log(2 * np.pi * cov**2 * np.e)
        self.cov = (np.exp(2 * entropy) / 2 * np.pi * np.e)**0.5
        
    
    
    
