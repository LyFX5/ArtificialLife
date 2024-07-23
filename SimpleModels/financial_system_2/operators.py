import numpy as np


def diagonal_reflection(n, axis) -> np.ndarray:
    I = np.diag(np.ones(n), k=0)
    I[axis, axis] = -1
    return I


def min_max_normalization(X) -> np.ndarray:
    X = X.copy()
    (n ,m) = X.shape
    for i in range(n):
        v = X[i, :]
        X[i, :] = (v - v.min())
        dif = v.max() - v.min()
        if dif != 0:
            X[i, :] /= dif
    for j in range(m):
        v = X[:, j]
        X[:, j] = (v - v.min())
        dif = v.max() - v.min()
        if dif != 0:
            X[:, j] /= dif
    return X

