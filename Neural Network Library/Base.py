import numpy as np
import scipy.linalg
import pickle

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
    """Combines bias vectors into weight matrices"""
    augmented_weights = []
    for i in range(len(weights)):
        augment = np.vstack((biases[i].T,weights[i]))
        augmented_weights.append(augment)
    return augmented_weights

def create_network(x, y, neurons, activations):
    """Creates the neural network dictionary"""
    weights = []; biases = []
    layers = len(neurons)
    for i in range(layers-1):
        next_layer = int(neurons[i+1])
        current_layer = int(neurons[i])
        weight = np.random.rand(current_layer,next_layer)*0.5
        bias = np.random.rand(next_layer, 1)*0.5
        weights.append(weight)
        biases.append(bias)

    weights = augment_network(weights, biases)
    nnet = {'Predictors': x,
            'Pass Forward': x.copy(),
            'Outcomes': y,
            'Outcomes_Subset': y.copy(),
            'Weights': weights,
            'Neurons': neurons,
            'Activations': activations}
    return nnet

def create_network2(x,y,layers,activations):

def load_nnet(path='output.pkl'):
    """Loads a neural network dictionary from a pickle file"""
    with open(path, 'rb') as pickle_file:
        loaded_data = pickle.load(pickle_file)
    nnet = loaded_data['nnet']
    return nnet

def matrix_norm(matrix):
    """Computes the matrix norm Trace(A'A)"""
    return np.trace(matrix.T @ matrix)

def mean_squared_error(y_predictions,y_outcomes):
    n = y_outcomes.shape[0]
    residuals = y_predictions-y_outcomes
    SSE = matrix_norm(residuals)
    MSE = SSE/n
    return MSE

def permutation_matrix(m, n):
    """Commutation Matrix: K vec(A) = vec(A.T)"""
    w = np.arange(m * n).reshape((m, n), order="F").T.ravel(order="F")
    return np.eye(m * n)[w, :]

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

def softmax(x):
    x = x - np.max(x)
    row_sum = np.sum(np.exp(x))
    return np.array([np.exp(x_i) / row_sum for x_i in x])

def softmax_batch(x):
    """This is the primary function for softmax"""
    row_maxes = np.max(x, axis=1)
    row_maxes = row_maxes[:, np.newaxis]  # for broadcasting
    x = x - row_maxes
    return np.array([softmax(row) for row in x])

def softmax_derivative(x):
    """This is the primary function for softmax derivatives"""
    n = x.shape[0]
    p = x.shape[1]
    blocks = softmax_batch_jacobian(x)
    J = scipy.linalg.block_diag(*blocks)
    Q = permutation_matrix(p,n)
    return Q @ J @ Q.T

def softmax_jacobian(s):
    return np.diag(s) - np.outer(s, s)

def softmax_batch_jacobian(x):
    return np.array([softmax_jacobian(row) for row in x])

def store_nnet(nnet, path='output.pkl'):
    """Saves neural network dictionary to a pickle file"""
    with open(path, 'wb') as pickle_file:
        pickle.dump({'nnet': nnet}, pickle_file)

def to_matrix(vector, matrix_dimension):
    """Converts a vector to a matrix with given dimensions"""
    matrix = vector.reshape(matrix_dimension, order="F")
    return matrix

def to_vector(matrix):
    """Converts matrix to a vector by stacking columns"""
    matrix_dimensions = matrix.shape
    vector_dimensions = np.prod(matrix_dimensions)
    vector = matrix.reshape((vector_dimensions,1), order="F")
    return vector