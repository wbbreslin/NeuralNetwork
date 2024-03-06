import numpy as np
import scipy.linalg
import pickle

def augment_predictor(x_predictors):
    """
    Adds a column of ones to a matrix
    Note: This function needs to be updated
    """
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
    """
    Creates the neural network dictionary
    x:              predictor variables (np array)
    y:              response variables (np array)
    neurons:        a list of integers for neurons in each layer
    activations:    a list of strings for names of activation functions
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
    act_first_derivatives = [fn.__name__ + "_derivative" for fn in activations]
    act_first_derivatives = [globals()[fn] for fn in act_first_derivatives]
    act_second_derivatives = [fn.__name__ + "_second_derivative" for fn in activations]
    act_second_derivatives = [globals()[fn] for fn in act_second_derivatives]
    weights = augment_network(weights, biases)
    nnet = {'Predictors': x,
            'Pass Forward': x.copy(),
            'Outcomes': y,
            'Outcomes_Subset': y.copy(),
            'Weights': weights,
            'Neurons': neurons,
            'Activations': activations,
            'Activation_Derivatives': act_first_derivatives,
            'Activation_Second_Derivatives': act_second_derivatives}
    return nnet

def indices_of_smallest_nonzero_k(arr, k):
    # Find the non-zero elements
    non_zero_values = arr[arr != 0]

    # Check if there are enough non-zero values to find the smallest k
    if len(non_zero_values) >= k:
        # Find the indices of the smallest k non-zero values
        indices_nonzero = np.argpartition(non_zero_values, k)[:k]

        # Get the corresponding non-zero values
        smallest_nonzero_values = non_zero_values[indices_nonzero]

        # Find the indices of these values in the original array
        indices = np.where(np.isin(arr, smallest_nonzero_values))

        return indices
    else:
        # If there are not enough non-zero values, raise an exception or handle it as needed
        raise ValueError("Not enough non-zero values to find the smallest k.")


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

def permutation_matrix_by_indices(U, V):
    """
    Returns a matrix that permutes rows and columns of a vector in any order based on U and V
    U:  A vector of indices
    V:  A vector of the same indicies, but in a different order
    """
    M = np.zeros((len(U), len(V)))
    for u, v in zip(U, V):
        M[u, v] = 1
    return M

def generate_random_indices(num_rows, random_seed=None):

    if random_seed is not None:
        np.random.seed(random_seed)

    random_indices = np.random.permutation(num_rows)
    return random_indices

def parameter_vector_to_weights(parameter_vector, weights):
    elements = [w.size for w in weights]
    partitions = np.append(0, np.cumsum(elements))
    weights_list = []
    for i in range(len(weights)):
        partition = parameter_vector[partitions[i]:partitions[i + 1]]
        partition = partition.reshape(weights[i].shape, order='F')
        weights_list.append(partition)
    return(weights_list)

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

def weights_to_parameter_vector(weights):
    vectorized_weights = [to_vector(w) for w in weights]
    vec = np.vstack(vectorized_weights)
    return(vec)

def unit_vector(i,n):
    v = np.zeros((n,1))
    v[i] = 1
    return v

def unit_matrix(i,n):
    m = np.zeros((n,n))
    m[i,i] = 1
    return(m)