import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x>0,1,0)

def relu_second_derivative(x):
    return 0*x

def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def sigmoid_second_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))*(1-2*sigmoid(x))

def linear(x):
    return(x)

def linear_derivative(x):
    return x*0+1

def linear_second_derivative(x):
    return x*0

x = np.array([[1,2,-1,1,-3]])
print(relu(x))
print(relu_derivative(x))