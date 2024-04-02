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
    return np.apply_along_axis(softmax_hessian, 0, x)

def softmax_jacobian(x):
    s = softmax(x)
    n = len(s)
    jac = -np.outer(s, s)
    jac[np.diag_indices(n)] = s * (1 - s)
    return jac

def softmax_hessian(x):
    """Compute the Hessian matrix of the softmax function."""
    s = softmax(x)
    n = len(s)
    jac_first = softmax_derivative(x)
    hessian = np.zeros((n, n, n))
    diag_idx = np.diag_indices(n)
    hessian[diag_idx] = s * (1 - 2 * s)
    hessian += np.einsum('i,jk->ijk', s, np.eye(n))
    hessian -= np.einsum('ij,k->ijk', jac_first, s)
    return hessian