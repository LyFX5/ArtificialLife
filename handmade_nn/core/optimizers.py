import numpy as np
from abc import ABC, abstractmethod
import pdb

    
class Optimizer(ABC):
    @abstractmethod
    def optimize(self, weights_matrix: np.ndarray, error_der_matrix: np.ndarray):
        pass

class OptimizerWithState(Optimizer):
    @abstractmethod
    def reboot(self):
        pass


class GradDesc(Optimizer):
    def __init__(self, learning_rate: np.float32):
        assert learning_rate > 0
        self.lr = learning_rate

    def optimize(self, weights_matrix: np.ndarray, error_der_matrix: np.ndarray):
        assert np.array_equal(weights_matrix.shape, error_der_matrix.shape)
        return weights_matrix - self.lr * error_der_matrix


class Momentum(OptimizerWithState):
    def __init__(self, learning_rate: np.float32, momentum: np.float32):
        assert learning_rate > 0
        assert 0 < momentum and momentum < 1 
        self.lr = learning_rate
        self.m = momentum
        self.change = 0

    def optimize(self, weights_matrix: np.ndarray, error_der_matrix: np.ndarray):
        assert np.array_equal(weights_matrix.shape, error_der_matrix.shape)
        self.change = self.m * self.change + self.lr * error_der_matrix
        return weights_matrix - self.change

    def reboot(self):
        self.change = 0


class Adagrad(OptimizerWithState):
    def __init__(self, learning_rate: np.float32, epsilon: np.float32 = 10**(-8)):
        assert epsilon * learning_rate > 0
        self.lr = learning_rate
        self.eps = epsilon
        self.accum = 0
    
    def optimize(self, weights_matrix: np.ndarray, error_der_matrix: np.ndarray):
        assert np.array_equal(weights_matrix.shape, error_der_matrix.shape)
        self.accum += np.power(error_der_matrix, 2)
        return weights_matrix - error_der_matrix * self.lr / (np.sqrt(self.accum) + self.eps)
         

    def reboot(self):
        self.accum = np.zeros_like(self.accum)


class RMSProp(OptimizerWithState):
    def __init__(self, learning_rate: np.float32, momentum: np.float32 = 0.9, epsilon: np.float32 = 10**(-8)):
        assert epsilon * learning_rate > 0
        assert 0 < momentum and momentum < 1 
        self.lr = learning_rate
        self.m = momentum
        self.eps = epsilon
        self.accum = 0
    
    def optimize(self, weights_matrix: np.ndarray, error_der_matrix: np.ndarray):
        assert np.array_equal(weights_matrix.shape, error_der_matrix.shape)
        self.accum = self.m * self.accum + (1 - self.m) * np.power(error_der_matrix, 2)
        return weights_matrix - error_der_matrix * self.lr / (np.sqrt(self.accum) + self.eps)
         
    def reboot(self):
        self.accum = np.zeros_like(self.accum)


class Adam(OptimizerWithState):
    def __init__(self,
                 learning_rate: np.float32,
                 momentum_1: np.float32 = 0.9, 
                 momentum_2: np.float32 = 0.9, 
                 epsilon: np.float32 = 10**(-8)):
        assert epsilon * learning_rate > 0
        assert 0 < momentum_1 and momentum_1 < 1 
        assert 0 < momentum_2 and momentum_2 < 1 
        self.step = 0
        self.lr = learning_rate
        self.m1 = momentum_1
        self.m2 = momentum_2
        self.eps = epsilon
        self.accum_1_ord = 0
        self.accum_2_ord = 0
    
    def optimize(self, weights_matrix: np.ndarray, error_der_matrix: np.ndarray):
        assert np.array_equal(weights_matrix.shape, error_der_matrix.shape)
        self.step += 1
        # Пересчет накопителей производных
        self.accum_1_ord = self.m1 * self.accum_1_ord + (1 - self.m1) * error_der_matrix
        self.accum_2_ord = self.m2 * self.accum_2_ord + (1 - self.m2) * np.power(error_der_matrix, 2)
        # Корректировка накопителей для несмещенности
        self.accum_1_ord_unbiased = self.accum_1_ord / (1 - self.m1 ** self.step)
        self.accum_2_ord_unbiased = self.accum_2_ord / (1 - self.m2 ** self.step)
        step = self.accum_1_ord_unbiased * self.lr / (np.sqrt(self.accum_2_ord_unbiased) + self.eps)
        return weights_matrix - step

    def reboot(self):
        self.step = 0
        self.accum_1_ord = np.zeros_like(self.accum_1_ord)
        self.accum_2_ord = np.zeros_like(self.accum_2_ord)
    