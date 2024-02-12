import numpy as np


def mean_squared_error(y1,y2):
    n = y1.shape[0]
    residuals = y1-y2
    MSE = np.trace(residuals.T @ residuals) / n
    return MSE
