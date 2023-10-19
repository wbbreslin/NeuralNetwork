import numpy as np

def compute_jacobian(func, x, epsilon=1e-6):
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

input_vector = np.array([2.0, 3.0])
jacobian_matrix = compute_jacobian(example_function, input_vector)
print("Jacobian matrix:")
print(jacobian_matrix)
