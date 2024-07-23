import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def calc(prediction: np.ndarray, target: np.ndarray):
        pass
    @abstractmethod
    def grad(outputs: np.ndarray, target: np.ndarray):
        pass


class Crossentropy(Loss):
    def calc(prediction: np.ndarray, target: np.ndarray, epsilon: np.float32 = 10**(-8)) -> np.float32:
        """prediction.shape == (batch_size, 1)"""
        indxs = np.arange(prediction.shape[0])
        return -np.log(prediction[indxs, target] + epsilon).sum() / indxs.size
    def grad(outputs: np.ndarray, target: np.ndarray, epsilon: np.float32 = 10**(-8)) -> np.ndarray:
        """outputs.shape == (batch_size, n_classes) target.shape == (batch_size, 1)
            return (batch_size, n_neurons)
        """
        indxs = np.arange(outputs.shape[0])
        error = np.zeros((outputs.shape[0], outputs.shape[1]))
        error[indxs, target] =  - 1 / (outputs[indxs, target] + epsilon)
        return error
