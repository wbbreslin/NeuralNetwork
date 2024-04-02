import numpy as np

def linear(x):
    return(x)

def linear_derivative(x):
    return x*0+1

def linear_second_derivative(x):
    return x*0

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
    ds = np.apply_along_axis(sigmoid_jacobian, 0, x)
    ds = np.transpose(ds, (2, 0, 1))
    return ds

def sigmoid_second_derivative(x):
    np.apply_along_axis(sigmoid_hessian, 0, x)

def sigmoid_jacobian(x):
    return np.diagflat(sigmoid(x) * (1 - sigmoid(x)))

def sigmoid_hessian(x):
    return sigmoid(x)*(1-sigmoid(x))*(1-2*sigmoid(x))

def softmax(x):
    return np.apply_along_axis(softmax_function, 0, x)
def softmax_function(x):
    return np.exp(x)/np.sum(np.exp(x))

def softmax_derivative(x):
    ds = np.apply_along_axis(softmax_jacobian, 0, x)
    ds = np.transpose(ds,(2,0,1))
    return(ds)
def softmax_second_derivative(x):
    d2s = np.apply_along_axis(softmax_hessian, 0, x)
    d2s = np.transpose(d2s, (3, 0, 1, 2))
    return d2s

def softmax_jacobian(x):
    s = softmax(x)
    n = len(s)
    jac = -np.outer(s, s)
    jac[np.diag_indices(n)] = s * (1 - s)
    return jac

def softmax_hessian(x):
    """Compute the Hessian matrix of the softmax function."""
    s = softmax_function(x)
    ds = softmax_jacobian(x)
    size = s.size
    i, j, k = np.indices((size, size, size))
    tensor = np.where((i == j) & (j == k), 1 - 2 * s[i], 0)
    tensor = np.where((j == k) & (i != j), -s[i], tensor)
    tensor = np.where((i == j) & (i != k), -s[k], tensor)
    hessian = ds @ tensor
    return hessian

'''
x = np.array([[2],
              [3]])

print("Softmax")
print(softmax(x))
print("Jacobian")
print(softmax_derivative(x))
print("Hessian")
print(softmax_second_derivative(x))

s = softmax(x)
g = softmax_derivative(x)
s1 = float(s[0])
s2 = float(s[1])
grad_g = np.array([[1-2*s1, -s2, -s2, 0],
                   [0, -s1, -s1, 1-2*s2]])

print(g @ grad_g)
'''
