import numpy as np
from abc import ABC, abstractmethod
from sklearn.metrics import(
    precision_score,
    recall_score,
    roc_auc_score
)



class Metric(ABC):
    @abstractmethod
    def calc(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass


class Precision(Metric):
    def __init__(self, average: str = 'macro'):
        self.params = {'average': average}

    def calc(self, y_true: np.ndarray, y_pred: np.ndarray):
        return precision_score(y_true, y_pred, **self.params)

class Recall(Metric):
    def __init__(self, average: str = 'macro'):
        self.params = {'average': average}

    def calc(self, y_true: np.ndarray, y_pred: np.ndarray):
        return recall_score(y_true, y_pred, **self.params)

class Roc_Auc(Metric):
    def __init__(self, average: str = 'macro', multi_class: str = 'ovo'):
        self.params = {'average': average, 'multi_class': multi_class}

    def calc(self, y_true: np.ndarray, y_pred: np.ndarray):
        return roc_auc_score(y_true, y_pred, **self.params)