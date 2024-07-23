import numpy as np
from typing import Tuple
from copy import deepcopy
from scipy.ndimage import correlate
from abc import ABC, abstractmethod
from .activations import Activation, ReLu
from .optimizers import Optimizer, Adam
from .initializers import Initializer
import pdb


def add_unit_h(source: np.ndarray):
    size = source.shape[0]
    return np.hstack((source, np.ones((size, 1))))

   
class Layer(ABC):
    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass
    @abstractmethod
    def backward(self,
                 inputs: np.ndarray,
                 outputs: np.ndarray,
                 error_grad_mat: np.ndarray, 
                 l1: np.float32 = 0.001, 
                 l2: np.float32 = 0.001) -> np.ndarray:
        pass


class Dense(Layer):
    def __init__(self,
                 size: int,
                 prev_size: int,
                 next_size: int,
                 initializer_class: type,
                 activation_class: type,
                 optimizer: Optimizer):
        assert issubclass(activation_class, Activation)
        assert issubclass(initializer_class, Initializer)
        self.size = size
        self.initializer = initializer_class
        self.activation = activation_class
        self.optimizer = optimizer
        self.weights = self.initializer.initialize(prev_size + 1, self.size, next_size) #(n_inputs, n_neurons)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """inputs: np.ndarray (batch_size, n_input)
            return shape == (batch_size, n_neurons)"""
        # Добавим к входным векторам единицы для учета веса смещения
        w_inputs = add_unit_h(inputs)
        return self.activation.calc(w_inputs.dot(self.weights))

    def backward(self,
                 inputs: np.ndarray, 
                 outputs: np.ndarray, 
                 error_grad_mat: np.ndarray, 
                 l1:np.float32 = 0.001, 
                 l2:np.float32 = 0.001) -> np.ndarray:
        """надо вернуть градиент ошибки по выходам предыдущего слоя и в текущем поменять веса
            inputs.shape == (batch_size, n_prev_neurons),
            outputs.shape == (batch_size, n_neurons),
            error_grad_mat.shape == (batch_size, n_neurons),
            return.shape == (batch_size, n_neurons)
        """
        batch_size = inputs.shape[0]
        # Добавим к входным векторам единицы для учета веса смещения
        w_inputs = add_unit_h(inputs)
        # Найдем градиент ошибки по входам функции активации
        grad_out_by_summator = self.activation.error_back_prop(outputs, error_grad_mat)
        # Найдем градиент ошибки по выходам предыдущего слоя
        grad_by_prev_layer = grad_out_by_summator.dot(self.weights[:-1, :].T)
        # Найдем матрицу производных ошибки по весам нейронов текущего слоя
        error_der_matrix = w_inputs.T.dot(grad_out_by_summator) / batch_size + 2 * l2 * self.weights + l1 * ((self.weights > 0) * 2 - 1)
        # Произведем шаг оптимизации весов по найенным производным
        self.weights = self.optimizer.optimize(self.weights, error_der_matrix)
        # Вернем градиент ошибки по выходам предыдущего слоя
        return grad_by_prev_layer


class BatchNormalizer(Layer):
    def __init__(self, size: np.int32, optimizer: Optimizer, epsilon: np.float32 = 10**(-8)):
        self.scale = np.ones((1, size), dtype=np.float32)
        self.bias = np.zeros((1, size), dtype=np.float32)        
        self.eps = epsilon
        self.optimizer_scale = optimizer
        self.optimizer_bias = deepcopy(optimizer)

    def forward(self, inputs: np.ndarray):
        """inputs: np.ndarray (batch_size, n_prev_neurons)
            return shape == (batch_size, n_neurons)"""
        self.mean = inputs.mean(axis=0, keepdims=True)
        self.norm = 1 / (inputs.std(axis=0, keepdims=True) + self.eps)
        return self.scale * self.norm *(inputs - self.mean)  + self.bias

    def backward(self, 
                 inputs: np.ndarray,
                 outputs: np.ndarray,
                 error_grad_mat: np.ndarray,
                 l1: np.float32 = 0.001,
                 l2: np.float32 = 0.001) -> np.ndarray:
        """надо вернуть градиент ошибки по выходам предыдущего слоя и в текущем поменять параметры масштаба и смещения
            inputs.shape == (batch_size, n_neurons),
            outputs.shape == (batch_size, n_neurons),
            error_grad.shape == (batch_size, n_neurons),
            return.shape == (batch_size, n_neurons)
        """
        batch_size = inputs.shape[0]
        # Найдем градиент ошибки по нормализованным входам
        grad_by_normalized = error_grad_mat * self.scale
        # Найдем градиент ошибки по дисперсии
        grad_by_disp = -0.5 * np.power(self.norm, 3) * (grad_by_normalized * (inputs - self.mean)).sum(axis=0, keepdims=True)
        # Найдем градиент ошибки по мат. ожиданию
        grad_by_mean = -self.norm * (grad_by_normalized).sum(axis=0, keepdims=True) -2 * grad_by_disp * (inputs - self.mean).sum(axis=0, keepdims=True) / (batch_size - 1)
        # Найдем градиент ошибки по выходам предыдущего слоя
        grad_by_prev_layer = self.norm * grad_by_normalized + grad_by_disp * 2 * (inputs - self.mean) / (batch_size - 1) + grad_by_mean / batch_size
        # Найдем градиент ошибки по коэфициенту сдвига
        error_grad_by_bias = error_grad_mat.sum(axis=0, keepdims=True)
        # Произведем шаг оптимизации коэффициента сдвига
        self.bias = self.optimizer_bias.optimize(self.bias, error_grad_by_bias)
        # Найдем градиент ошибки по коэфициенту масштаба        
        error_grad_by_scale = (self.norm * (inputs - self.mean) * error_grad_mat).sum(axis=0, keepdims=True)
        # Произведем шаг оптимизации коэффициента масштаба
        self.scale = self.optimizer_scale.optimize(self.scale, error_grad_by_scale)
        # Вернем градиент ошибки по выходам предыдущего слоя
        return grad_by_prev_layer


class Flatten(Layer):
    def __init__(self):
        pass

    def forward(self, inputs: np.ndarray):
        """inputs: np.ndarray (batch_size, rows, cols)
           return shape == (batch_size, rows * cols)"""
        return inputs.reshape((inputs.shape[0], -1))

    def backward(self,
                 inputs: np.ndarray,
                 outputs: np.ndarray,
                 error_grad_mat: np.ndarray,
                 l1: np.float32 = 0.001,
                 l2: np.float32 = 0.001) -> np.ndarray:
        """
        Надо изменить рамерность матрицы градиента ошибки
        error_grad_mat.shape == (batch_size, n_elements),
        return:
            np.ndarray(inputs.shape)
        """
        return error_grad_mat.reshape(inputs.shape)


class MaxPool1D(Layer):
    def __init__(self, pool_size: int=2, stride: int=1, axis: int=1):
        self.pool_size = pool_size
        self.stride = stride
        self.axis = axis
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """inputs: np.ndarray(batch_size, *dimensions)"""
        pooled = np.apply_along_axis(
            func1d=self.__pool,
            axis=self.axis,
            arr=inputs
        )
        result = np.take(pooled, indices=0, axis=self.axis)
        self.max_inds = np.take(pooled, indices=1, axis=self.axis).astype(np.uint32)
        return result
    
    def __pool(self, inputs: np.ndarray) -> np.ndarray:
        """inputs: 1d array"""
        output_size = np.floor((inputs.size - self.pool_size + 1) / self.stride + 0.5).astype(int)
        result = np.zeros(output_size, dtype=np.float32)
        max_inds = np.zeros_like(result, dtype=np.uint32)
        for out_ind in range(output_size):
            start = out_ind * self.stride
            stop = start + self.pool_size
            max_inds[out_ind] = np.argmax(inputs[start: stop]) + start
            result[out_ind] = inputs[max_inds[out_ind]]
        return result, max_inds
    
    def backward(self,
                 inputs: np.ndarray,
                 outputs: np.ndarray,
                 error_grad_mat: np.ndarray,
                 l1: np.float32 = 0.001,
                 l2: np.float32 = 0.001) -> np.ndarray:
        error_by_inputs = np.zeros_like(inputs)
        indexes = np.array(list(np.ndindex(*(error_grad_mat.shape)))) 
        np.put(indexes, np.arange(indexes.shape[0]) * indexes.shape[1] + self.axis, self.max_inds.flatten())
        error_by_inputs[tuple(indexes.T)] = error_grad_mat.flatten()
        return error_by_inputs


class MaxPool2D(Layer):
    def __init__(self,
                 pool_sizes: Tuple[int, int]=(2, 2),
                 strides: Tuple[int, int]=(1, 1),
                 axes: Tuple[int, int]=(1, 2)):
        assert axes[0] < axes[1]
        self.pool_sizes = pool_sizes
        self.strides = strides
        self.axes = list(axes)
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """inputs: np.ndarray(batch_size, *dimensions)"""
        if len(inputs.shape) == 2:
            result, self.max_inds = self.__pool(inputs)
        else:
            result_dims = np.array(list(inputs.shape))
            result_dims[self.axes] = self.__calc_out_size(result_dims[self.axes]) 
            result = np.empty(result_dims, dtype=np.float32)
            self.max_inds = np.zeros_like(result, dtype=object)
            poped_axes = list(range(len(inputs.shape)))
            poped_axes.pop(self.axes[0])
            poped_axes.pop(self.axes[1] - 1)
            poped_axes = tuple(np.expand_dims(poped_axes, 0))
            poped_dims = np.array(inputs.shape)[poped_axes]
            indexes = np.array([slice(None)] * len(inputs.shape), dtype=object)
            for others in np.ndindex(*poped_dims):
                indexes[poped_axes] = others
                tupled = tuple(indexes)
                result[tupled], self.max_inds[tupled] = self.__pool(inputs[tupled]) 
        return result
    
    def __calc_out_size(self, inputs_shape: tuple):
        """inputs_shape: shape of pooled 2d matrix"""
        return [
            np.floor((inputs_shape[i] - self.pool_sizes[i] + 1) / self.strides[i] + 0.5).astype(int)
            for i in range(2)
        ]

    def __pool(self, inputs: np.ndarray) -> np.ndarray:
        """inputs: 2d array"""
        output_sizes = self.__calc_out_size(inputs.shape)
        result = np.zeros(output_sizes, dtype=np.float32)
        max_inds = np.zeros(result.shape, dtype=tuple)
        for out_inds in np.ndindex(*output_sizes):
            start = [out_inds[i] * self.strides[i] for i in range(2)]
            stop = [start[i] + self.pool_sizes[i] for i in range(2)]
            borders = tuple([slice(start[i], stop[i]) for i in range(2)])
            max_inds[out_inds] = tuple(np.unravel_index(np.argmax(inputs[borders]), self.pool_sizes) + np.array(start))
            result[out_inds] = inputs[max_inds[out_inds]]
        return result, max_inds
    
    def backward(self,
                inputs: np.ndarray,
                outputs: np.ndarray,
                error_grad_mat: np.ndarray,
                l1: np.float32 = 0.001,
                l2: np.float32 = 0.001) -> np.ndarray:
        error_by_inputs = np.zeros_like(inputs)
        flattend_max = np.array(list(self.max_inds.flatten()))
        indexes = np.array(list(np.ndindex(*(error_grad_mat.shape)))) 
        for i in range(2):
            np.put(indexes, np.arange(indexes.shape[0]) * indexes.shape[1] + self.axes[i], flattend_max[:, i])
        error_by_inputs[tuple(indexes.T)] = error_grad_mat.flatten()
        return error_by_inputs


class Conv2D(Layer):
    def __init__(self,
                 kernels_number: np.uint8=1,
                 kernel_shape: Tuple[int, int]=(3, 3),
                 input_channels: np.uint8=1,
                 activation_class: type=ReLu,
                 optimizer: Optimizer=Adam(10**(-3), momentum_1=0.9, momentum_2=0.9)):
        """
        Свёрточный слой без аггрегации слоев для ядра
        :param kernels_number: np.uint8
            Количество ядер свертки
        :param kernel_shape: Tuple[int, int]
            Размерность ядра свертки
        :param input_channels: np.uint8
            Количество каналов входного тензора (в каждом пакете)
        :param activation_class: type
            Класс функции активации для элементов выходных тензоров
        :param optimizer: Optimizer
            Оптимизатор для обучения коэффициентов ядер свертки 
        """
        assert issubclass(activation_class, Activation)
        assert kernels_number > 0 and len(kernel_shape) == 2 and input_channels > 0
        self.activation = activation_class
        self.optimizer = optimizer
        self.optimizer_bias = deepcopy(optimizer)
        self.kernels_number = kernels_number
        self.kernel_shape = kernel_shape
        self.input_channels = input_channels
        self.kernels = np.random.rand(kernels_number, *kernel_shape, input_channels) * 2 - 1
        self.biases = np.random.rand(kernels_number, 1, 1, input_channels) * 2 - 1
        self.kernels_der = np.zeros_like(self.kernels)  # матрица производных функции потерь по элементам ядер свертки
        self.biases_der = np.zeros_like(self.biases)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Свёртка каналов с соответствующими каналами ядер  
        :params inputs: np.ndarray (batch_size, rows, columns, channels)
        :return: np.ndarray (batch_size, rows, cols, filters)
        """
        output = np.zeros(shape=(inputs.shape[0],
                                 inputs.shape[1],
                                 inputs.shape[2],
                                 self.kernels_number),
                         dtype=np.float32)
        for batch, channel, kernel in np.ndindex(inputs.shape[0], inputs.shape[-1], self.kernels_number):
            output[batch, :, :, kernel] += correlate(
                inputs[batch, :, :, channel],
                self.kernels[kernel, :, :, channel],
                mode="constant",
                cval=0
            ) + self.biases[kernel, :, :, channel]
        return self.activation.calc(output)

    def backward(self,
                 inputs: np.ndarray, 
                 outputs: np.ndarray, 
                 error_grad_mat: np.ndarray, 
                 l1:np.float32 = 0.001, 
                 l2:np.float32 = 0.001) -> np.ndarray:
        """
        Обновить веса и вернуть матрицу частных производных по матрице входных значений
        :params inputs: np.ndarray (batch_size, rows, columns, channels)
            Матрица входных значений, полученная во время прямого распространения
        :params outputs: np.ndarray (batch_size, rows, columns, filters)
            Матрица выходных значений, полученная во время прямого распространения
        :params error_grad_mat: np.ndarray (batch_size, rows, columns, filters)
            Матрица частных производных функции потерь по элементам выхода
        """
        batch_size = inputs.shape[0]
        wided_rows = inputs.shape[1] + self.kernel_shape[0] - 1
        wided_cols = inputs.shape[2] + self.kernel_shape[1] - 1
        rows_radius = (self.kernel_shape[0] - 1) //  2
        cols_radius = (self.kernel_shape[1] - 1) //  2
        # Создадим расширенную матрицу входных значений на радиусы ядра светки
        divided_inputs = np.zeros((batch_size, wided_rows, wided_cols, inputs.shape[-1]), dtype=np.float32)
        # Разделим матрицу входных значений на размер пакета для накопления ошибок в усредненном виде
        divided_inputs[:, rows_radius: -rows_radius, cols_radius: -cols_radius, :] = inputs / batch_size
        # Найдем градиент ошибки по входам функции активации
        grad_error_by_act_in = self.activation.error_back_prop(outputs, error_grad_mat)
        # Матрица частных производных функции потерь по элементам входной матрицы
        grad_by_prev_layer = np.zeros_like(inputs, dtype=np.float32)
        for batch, channel, kernel in np.ndindex(inputs.shape[0], inputs.shape[-1], self.kernels_number):
            grad_by_prev_layer[batch, :, :, channel] += correlate(
                grad_error_by_act_in[batch, :, :, kernel],
                np.flip(self.kernels[kernel, :, :, channel], axis=(0,1)),
                mode="constant",
                cval=0
            )
        # Матрица частных производных функции потерь по элементам ядер свёртки
        self.kernels_der.fill(0)
        for batch, channel, kernel in np.ndindex(inputs.shape[0], inputs.shape[-1], self.kernels_number):
            for row, col in np.ndindex(*self.kernel_shape):
                stop_row = wided_rows - (self.kernel_shape[0] - (row + 1))
                stop_col = wided_cols - (self.kernel_shape[1] - (col + 1))
                self.kernels_der[kernel, row, col, channel] = (
                    divided_inputs[batch, row: stop_row, col: stop_col, channel]
                     * grad_error_by_act_in[batch, :, :, kernel]
                ).sum()
        # Матрица частных произвдных функции потерь по смещениям сверток для результатов сверток
        self.biases_der.fill(0)
        for kernel in range(self.kernels_number):
            self.biases_der[kernel, :, :, :] = grad_error_by_act_in[:, :, :, kernel].sum()
        # Проведем обнвление коэффициентов ядер
        self.kernels = self.optimizer.optimize(self.kernels, self.kernels_der)
        # Проведём обновление вектора смещений
        self.biases = self.optimizer_bias.optimize(self.biases, self.biases_der)
        return grad_by_prev_layer
