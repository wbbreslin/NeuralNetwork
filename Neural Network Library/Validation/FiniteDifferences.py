import numpy as np
import Base as base
import FirstOrderModel as fom


def jacobian(func, x, epsilon=1e-6):
    """
    Compute the Jacobian matrix of a vector-valued function using finite differences.

    Parameters:
        func: A callable function that takes a 1D NumPy array (input vector) and returns a 1D or 2D array.
        x: The input vector at which to compute the Jacobian.
        epsilon: Small perturbation for finite differences.

    Returns:
        J: The Jacobian matrix (2D NumPy array).
    """
    n = len(x)
    m = len(func(x))
    J = np.zeros((m, n))

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon

        J[:, i] = (func(x_plus) - func(x_minus)) / (2 * epsilon)

    return J

# Example usage:
def example_function(x):
    y1 = x[0] ** 2
    y2 = x[0] * x[1]
    return np.array([y1, y2])

#input_vector = np.array([2.0, 3.0])
#jacobian_matrix = compute_jacobian(example_function, input_vector)
#print("Jacobian matrix:")
#print(jacobian_matrix)

"""The data set of predictor variables"""
x_predictors = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],
                        [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]]).T

"""The data set of outcome variables"""
y_outcomes = np.array([[1,1,1,1,1,0,0,0,0,0],
                      [0,0,0,0,0,1,1,1,1,1]]).T

W0 = np.array([[0.1, 0.2],
               [0.3, 0.4],
               [0.5, 0.6]])

W1 = np.array([[0.7, 0.8, 0.9],
               [1.0, 1.1, 1.2],
               [1.3, 1.4, 1.5]])

W2 = np.array([[1.6, 1.7],
               [1.8, 1.9],
               [2.0, 2.1],
               [2.2, 2.3]])

weights = [W0, W1, W2]

w0, d0 = base.to_vector(W0)
w1, d1 = base.to_vector(W1)
w2, d2 = base.to_vector(W2)

W = np.vstack((w0,w1,w2))

def jacobian_matrix(matrix, layer, weights, epsilon, x_predictors, y_outcomes):
    vector, dims = base.to_vector(matrix)
    n = len(vector)
    m = 1 #cost function is scalar
    J = np.zeros(n)

    for i in range(n):
        vector_plus = vector.copy()
        vector_minus = vector.copy()
        vector_plus[i] += epsilon
        vector_minus[i] -= epsilon
        matrix_plus = base.to_matrix(vector_plus, dims)
        matrix_minus = base.to_matrix(vector_minus, dims)

        weights_minus = weights.copy()
        weights_plus = weights.copy()
        weights_minus[layer] = matrix_minus
        weights_plus[layer] = matrix_plus

        nnet_plus = fom.forward_pass(x_predictors,
                                     weights_plus)

        nnet_minus = fom.forward_pass(x_predictors,
                                      weights_minus)

        y_plus = nnet_plus[0][-1]
        y_minus = nnet_minus[0][-1]

        cost_minus = base.mean_squared_error(y_minus,y_outcomes)
        cost_plus = base.mean_squared_error(y_plus,y_outcomes)

        J[i] = (cost_plus - cost_minus) / (2 * epsilon)

    G = base.to_matrix(J.T, dims)
    return G

gradient = jacobian_matrix(W0, 0, weights, 10**-2, x_predictors, y_outcomes)
print(gradient)