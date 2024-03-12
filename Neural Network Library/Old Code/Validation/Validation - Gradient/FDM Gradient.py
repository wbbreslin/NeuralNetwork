import numpy as np
import FirstOrderModel as fom
import copy
import Base as base

"""The data set of predictor variables"""
x_predictors = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],
                        [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]]).T

"""The data set of outcome variables"""
y_outcomes = np.array([[1,1,1,1,1,0,0,0,0,0],
                      [0,0,0,0,0,1,1,1,1,1]]).T

'''Define the initial weights'''
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

'''Create a neural network object'''
nnet = base.create_network(x_predictors,
                           y_outcomes,
                           neurons = [2,2,3,2],
                           activations = ["sigmoid","sigmoid","sigmoid","sigmoid"])

nnet['Weights']=weights

'''Define the function we are differentiating'''
def cost(y1,y2):
    residual = y1-y2
    loss = np.trace(residual@residual.T)/2
    return loss

'''Set up the gradient matrices'''
g0 = np.zeros(w0.shape)
g1 = np.zeros(w1.shape)
g2 = np.zeros(w2.shape)
gradients = [g0, g1, g2]

'''A copy of the neural network to modify'''
nnet_plus = copy.deepcopy(nnet)
nnet_minus = copy.deepcopy(nnet)
#nnet2['Weights'][0] = nnet2['Weights'][0]*2
#print(nnet['Weights'][0])
#print(nnet2['Weights'][0])

epsilon = 10**-6

for w in range(len(nnet['Weights'])):
    for row in range(nnet['Weights'][w].shape[0]):
        for col in range(nnet['Weights'][w].shape[1]):
            # Create separate copies of the weights for this iteration
            #weights_minus = [w.copy() for w in nnet['Weights']]
            #weights_plus = [w.copy() for w in nnet2['Weights']]
            nnet_plus = copy.deepcopy(nnet)
            nnet_minus = copy.deepcopy(nnet)

            lo = nnet_minus['Weights'][w][row, col] - epsilon
            hi = nnet_plus['Weights'][w][row, col] + epsilon
            nnet_minus['Weights'][w][row, col] = lo
            nnet_plus['Weights'][w][row, col] = hi

            # Compute the network outputs with the updated weights
            nnet_plus = fom.forward_pass(nnet_plus)
            nnet_minus = fom.forward_pass(nnet_minus)
            y_plus = nnet_plus['States'][-1]
            y_minus = nnet_minus['States'][-1]

            # Compute cost for plus and minus weights
            cost_minus = cost(y_minus, y_outcomes)
            cost_plus = cost(y_plus, y_outcomes)

            # Compute the gradient using finite differences
            delta = (cost_plus - cost_minus) / (2 * epsilon)

            gradients[w][row, col] = delta

print(gradients[0])