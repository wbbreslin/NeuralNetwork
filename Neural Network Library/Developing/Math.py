import numpy as np

def MSE(y_predictions,y_outcomes):
    n = y_outcomes.shape[0]
    residuals = y_predictions-y_outcomes
    SSE = matrix_norm(residuals)
    MSE = SSE/n
    return MSE
def matrix_norm(matrix):
    """Computes the matrix norm Trace(A'A)"""
    return np.trace(matrix.T @ matrix)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

def SSE(y_predictions,y_outcomes):
    residuals = y_predictions-y_outcomes
    SSE = matrix_norm(residuals)
    return SSE