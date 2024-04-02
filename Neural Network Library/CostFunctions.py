import numpy as np
import Base as base
import ActivationFunctions as af
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

def cross_entropy(y1, y2, s=1):
    epsilon = 1e-15  # to prevent log(0)
    y1 = np.clip(y1, epsilon, 1 - epsilon) # Clip the predicted values to prevent log(0)
    ce_loss = -np.sum(y2 * np.log(y1), axis=0)
    total_loss = np.sum(ce_loss)
    return total_loss
