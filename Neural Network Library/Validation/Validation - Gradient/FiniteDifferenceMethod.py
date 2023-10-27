import numpy as np
import FirstOrderModelOld as fom
# Need to fix this to fit into the new FOM framework

"""
Description:
Finite-difference approximation of the gradient for a neural network

Dataset: 
SIAM 2019
"""


"""The data set of predictor variables"""
x_predictors = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],
                        [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]]).T

"""The data set of outcome variables"""
y_outcomes = np.array([[1,1,1,1,1,0,0,0,0,0],
                      [0,0,0,0,0,1,1,1,1,1]]).T

"""Cost function"""
def cost(y1,y2):
    residual = y1-y2
    loss = np.trace(residual@residual.T)/2
    return loss

w0 = np.array([[0.1, 0.2],
               [0.3, 0.4],
               [0.5, 0.6]])

w1 = np.array([[0.7, 0.8, 0.9],
               [1.0, 1.1, 1.2],
               [1.3, 1.4, 1.5]])

w2 = np.array([[1.6, 1.7],
               [1.8, 1.9],
               [2.0, 2.1],
               [2.2, 2.3]])

weights = [w0, w1, w2]

epsilon = 10**-6
weights_minus = weights.copy()
weights_plus = weights.copy()

g0 = np.zeros(w0.shape)
g1 = np.zeros(w1.shape)
g2 = np.zeros(w2.shape)
gradients = [g0, g1, g2]

for w in range(len(weights)):
    for row in range(weights[w].shape[0]):
        for col in range(weights[w].shape[1]):
            # Create separate copies of the weights for this iteration
            weights_minus = [w.copy() for w in weights]
            weights_plus = [w.copy() for w in weights]

            lo = weights_minus[w][row, col] - epsilon
            hi = weights_plus[w][row, col] + epsilon
            weights_minus[w][row, col] = lo
            weights_plus[w][row, col] = hi

            # Compute the network outputs with the updated weights
            nnet_plus = fom.forward_pass(x_predictors, weights_plus)
            nnet_minus = fom.forward_pass(x_predictors, weights_minus)
            y_plus = nnet_plus[0][-1]
            y_minus = nnet_minus[0][-1]

            # Compute cost for plus and minus weights
            cost_minus = cost(y_minus, y_outcomes)
            cost_plus = cost(y_plus, y_outcomes)

            # Compute the gradient using finite differences
            delta = (cost_plus - cost_minus) / (2 * epsilon)

            gradients[w][row, col] = delta

print(gradients[0])