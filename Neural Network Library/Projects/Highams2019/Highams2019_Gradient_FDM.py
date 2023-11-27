import Highams2019 as data
import numpy as np
import FirstOrderModel as fom
import TrainingAlgorithms as train
import copy

'''Import the neural network and the data set'''
nnet = data.nnet
nnet = train.gradient_descent(nnet, max_iterations=3999)

'''Define the function we are differentiating'''
def cost(y1,y2):
    residual = y1-y2
    loss = np.trace(residual@residual.T)/2
    return loss

'''Set up the gradient matrices'''
g0 = np.zeros(nnet["Weights"][0].shape)
g1 = np.zeros(nnet["Weights"][1].shape)
g2 = np.zeros(nnet["Weights"][2].shape)
gradients = [g0, g1, g2]

'''A copy of the neural network to modify'''
nnet_plus = copy.deepcopy(nnet)
nnet_minus = copy.deepcopy(nnet)

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
            cost_minus = cost(y_minus, data.y_outcomes)
            cost_plus = cost(y_plus, data.y_outcomes)

            # Compute the gradient using finite differences
            delta = (cost_plus - cost_minus) / (2 * epsilon)

            gradients[w][row, col] = delta

print(gradients[0])