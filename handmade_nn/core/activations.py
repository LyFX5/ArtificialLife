import numpy as np
import pdb
from abc import ABC, abstractmethod


class Activation(ABC):
    @abstractmethod
    def calc(inputs: np.ndarray) -> np.ndarray:
        """
        Функция расчета значения функции активации для каждого элемента
        :params inputs: np.ndarray(batch_size, *signal.shape)
            Тензор входных значений
        :return: np.ndarray(batch_size, *signal.shape)
            Тензор выходных значений
        """
        pass
    @abstractmethod
    def error_back_prop(outputs: np.ndarray, error_grad: np.ndarray) -> np.ndarray:
        """
        Функция рассчета матрицы частных производных выходных значений функции активации по входным аргументам
        :params outputs: np.ndarray(batch_size, *signal.shape)
            Тезнор выходных значений, полученный при распространении вперёд
        :params error_grad: np.ndarray(batch_size, *signal.shape)
            Тезнор частных призводных функции потерь по выходным значениям функции активации
        :return: np.ndarray(batch_size, *signal.shape)
            Матрица частных производных функции потерь по входам функции активации
        """
        pass
        

class ReLu(Activation):
    @staticmethod
    def calc(inputs: np.ndarray) -> np.ndarray:
        return (inputs > 0) * inputs
    @staticmethod
    def error_back_prop(outputs: np.ndarray, error_grad: np.ndarray) -> np.ndarray:
        assert np.array_equal(outputs.shape, error_grad.shape) 
        return (outputs > 0) * error_grad


class Tanh(Activation):
    @staticmethod
    def calc(inputs: np.ndarray) -> np.ndarray:
        return np.tanh(inputs)
    @staticmethod
    def error_back_prop(outputs: np.ndarray, error_grad: np.ndarray) -> np.ndarray:
        assert np.array_equal(outputs.shape, error_grad.shape) 
        derivative = 1 - np.power(outputs, 2)
        return  derivative * error_grad


class Softmax(Activation):
    @staticmethod
    def calc(inputs: np.ndarray, epsilon: np.float32 = 10**(-8)) -> np.ndarray:
        assert len(inputs.shape) == 2        
        maximum = inputs.max(axis=1, keepdims=True)
        ln_of_sum = maximum + np.log(np.exp(inputs - maximum).sum(axis=1, keepdims=True) + epsilon)
        ln_result = inputs - ln_of_sum
        return np.exp(ln_result)
    @staticmethod
    def error_back_prop(outputs: np.ndarray, error_grad: np.ndarray) -> np.ndarray:
        assert len(outputs.shape) == 2
        assert np.array_equal(outputs.shape, error_grad.shape) 
        batch_size, n_neurons = outputs.shape
        der_matrix = np.empty(shape=(batch_size, n_neurons, n_neurons), dtype=np.float32)  # накопитель для матрицы производных функции активации по выходам сумматора
        der_matrix[:, range(n_neurons), range(n_neurons)] = (outputs - np.power(outputs, 2))  # заполнение главной диагонали
        rows, columns = np.triu_indices(n=n_neurons, k=1)  # индекс строк и столбцов верхнетреугольным матриц для каждого элемента в пакете
        triag_der_values = - outputs[:, rows] * outputs[:, columns]  # внедиагональные значения производной
        der_matrix[:, rows, columns] = triag_der_values
        der_matrix[:, columns, rows] = triag_der_values
        return (der_matrix * np.expand_dims(error_grad,1)).sum(axis=2)

class Linear(Activation):
    @staticmethod
    def calc(inputs: np.ndarray) -> np.ndarray:
        return inputs
    @staticmethod
    def error_back_prop(outputs: np.ndarray, error_grad: np.ndarray) -> np.ndarray:
        assert np.array_equal(outputs.shape, error_grad.shape) 
        return error_grad
        