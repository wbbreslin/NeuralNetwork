import numpy as np


def MSE(y1,y2):
    n = y1.shape[0]
    residuals = y1-y2
    MSE = np.trace(residuals.T @ residuals) / n
    return MSE

def half_SSE(y1,y2):
    n = y1.shape[0]
    residuals = y1-y2
    MSE = np.trace(residuals.T @ residuals) / 2
    return MSE



