import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return 1 if relu(x) > 0 else 0
def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def sigmoid_second_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))*(1-2*sigmoid(x))