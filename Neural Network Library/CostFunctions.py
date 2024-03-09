import numpy as np
import Base as base

def MSE(y1,y2,s=1):
    n = y1.shape[0]
    residuals = y1 - y2
    squared_residuals = (residuals @ residuals.T) * s
    MSE = np.trace(squared_residuals) / n
    return MSE

def half_SSE(y1,y2,s=1):
    residuals = y1 - y2
    squared_residuals = (residuals @ residuals.T) * s
    SSE = np.trace(squared_residuals) / 2
    return SSE