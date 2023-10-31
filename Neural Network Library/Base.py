import numpy as np
#import scipy as sp
#need to use scipy sparse.diags(vector) to save memory on Sigma''
def augment_predictor(x_predictors):
    dimensions = x_predictors.shape
    A1 = np.zeros((dimensions[1],1))
    A2 = np.eye(dimensions[1])
    A = np.hstack((A1,A2))
    B1 = np.ones((dimensions[0],1))
    B2 = np.zeros(dimensions)
    B = np.hstack((B1,B2))
    Z = x_predictors @ A + B
    return Z, A, B

def augment_network(weights, biases):
    """
    Description
    --------------------
    Combines bias vectors into weight matrices

    Inputs
    --------------------
    weights             : List of weight matrices
    biases              : List of bias vectors

    Outputs
    --------------------
    augmented_weights   : List of weight matrices with bias term absorbed into the matrix
    constants           : List of constants modifiers to account for bias absorption
    """

    augmented_weights = []
    for i in range(len(weights)):
        augment = np.vstack((biases[i].T,weights[i]))
        augmented_weights.append(augment)
    return augmented_weights

def communication_matrix(m, n):
    # determine permutation applied by K
    w = np.arange(m * n).reshape((m, n), order="F").T.ravel(order="F")

    # apply this permutation to the rows (i.e. to each column) of identity matrix and return result
    return np.eye(m * n)[w, :]

def create_network(neurons):
    """
    Description
    --------------------
    Initiates NN parameters given a list of neurons per layer

    Inputs
    --------------------
    neurons             : List of the number of neurons per layer e.g. [2,2,3,2]

    Outputs
    --------------------
    weights             : List of random weight matrices
    biases              : List of random bias vectors
    """

    weights = []; biases = []
    layers = len(neurons)
    for i in range(layers-1):
        next_layer = int(neurons[i+1])
        current_layer = int(neurons[i])
        weight = np.random.rand(current_layer,next_layer)*0.5
        bias = np.random.rand(next_layer, 1)*0.5
        weights.append(weight)
        biases.append(bias)
    return weights, biases

def matrix_norm(matrix):
    """
    Description
    --------------------
    Computes the matrix norm Trace(A'A)

    Inputs
    --------------------
    matrix              : Any matrix (m x n)

    Outputs
    --------------------
    norm                : A scalar value, the matrix norm
    """
    norm = np.trace(matrix.T @ matrix)
    return norm

def mean_squared_error(y_predictions,y_outcomes):
    """
    Description
    --------------------
    Computes the mean-squared error for a given set of predictions

    Inputs
    --------------------
    y_predictions       : A matrix (n x q) containing n data points in q variables
    y_outcomes          : A matrix (n x q) containing n data points in q variables

    Outputs
    --------------------
    MSE                 : A scalar value, mean-squared error
    """

    n = y_outcomes.shape[0]
    residuals = y_predictions-y_outcomes
    SSE = matrix_norm(residuals)
    MSE = SSE/n
    return MSE

def sigmoid(x):
    """
    Description
    --------------------
    Computes the sigmoid function component-wise

    Inputs
    --------------------
    x                   : A scalar, vector, or matrix

    Outputs
    --------------------
    y                   : Sigmoid of x
    """
    y = 1/(1+np.exp(-x))
    return y

def sigmoid_derivative(x):
    """
    Description
    --------------------
    Computes the sigmoid function component-wise
    NEED TO GENERALIZE THIS FUNCTION, ONLY WORKS FOR VECTOR & MATRIX INPUTS

    Inputs
    --------------------
    x                   : A scalar, vector, or matrix

    Outputs
    --------------------
    y                   : Sigmoid derivative of x

    Notes
    --------------------
    If x is a scalar, y will be a vector
    If x is a vector, y will be a matrix
    If x is a matrix, y will be a tensor (a higher-dimensional matrix)
    """
    #x, dims = to_vector(x)
    #y = np.diagflat(sigmoid(x)*(1-sigmoid(x)))
    y = sigmoid(x) * (1 - sigmoid(x))
    return y

def sigmoid_second_derivative(x):
    """
    Description
    --------------------
    Computes the sigmoid function component-wise
    NEED TO FIX THIS FUNCTION

    Inputs
    --------------------
    x                   : A scalar, vector, or matrix

    Outputs
    --------------------
    y                   : Sigmoid derivative of x

    Notes
    --------------------
    If x is a scalar, y will be a matrix
    If x is a vector, y will be a matrix
    If x is a matrix, y will be a tensor (a higher-dimensional matrix)
    """
    y = sigmoid(x)*(1-sigmoid(x))*(1-2*sigmoid(x))
    return y

def to_matrix(vector, matrix_dimension):
    """
    Description
    --------------------
    Converts a vector to a matrix with the given dimensions

    Inputs
    --------------------
    vector              : An (nm x 1) vector
    matrix_dimensions   : The original matrix dimensions (n x m)

    Outputs
    --------------------
    matrix              : An (n x m) matrix
    """

    matrix = vector.reshape(matrix_dimension, order="F")
    return matrix


def to_vector(matrix):
    """
    Description
    --------------------
    Converts a matrix to a vector, storing original dimensions

    Inputs
    --------------------
    matrix              : An (n x m) matrix

    Outputs
    --------------------
    vector              : An (nm x 1) vector
    matrix_dimensions   : The original matrix dimensions (n x m)
    """

    matrix_dimensions = matrix.shape
    vector_dimensions = np.prod(matrix_dimensions)
    vector = matrix.reshape((vector_dimensions,1), order="F")

    return vector, matrix_dimensions

