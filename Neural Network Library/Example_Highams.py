import Base as base
import NeuralNetwork as nn
import NNET as net2
from Data import data
import numpy as np
import ActivationFunctions as af
import CostFunctions as cf
import copy
import matplotlib.pyplot as plt

np.random.seed(100)

x = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],
                        [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]])

"""The data set of outcome variables"""
y = np.array([[1,1,1,1,1,0,0,0,0,0],
                      [0,0,0,0,0,1,1,1,1,1]])

nnet = nn.neural_network(layers=[2,2,3,2],
                           activation_functions = [af.sigmoid,
                                                   af.sigmoid,
                                                   af.softmax],
                           cost_function=cf.half_SSE)

weights = nnet.weights

training = data(x,y)
nnet.forward(training)
nnet.backward(training)
nnet.train(training, max_iterations =6000, step_size=0.25)
nnet.train(training, max_iterations =4000, step_size=0.05)

'''Make copies of trained network for FDM'''
nnet_FDM = copy.deepcopy(nnet)

'''Compute Hessian Matrix'''
nnet.compute_hessian()
hessian_SOA = nnet.hessian_matrix

'''Finite-Difference Approximation of the Hessian'''
nnet_plus = copy.deepcopy(nnet_FDM)
nnet_minus = copy.deepcopy(nnet_FDM)
epsilon = 10 ** -6

Hessians = []
for u in range(len(nnet.weights)):
    for w in range(len(nnet.weights)):
        shape_num = nnet.weights[u].shape
        shape_den = nnet.weights[w].shape
        p1 = shape_num[0]
        q1 = shape_num[1]
        p2 = shape_den[0]
        q2 = shape_den[1]
        Hessian = np.zeros((p1 * q1, p2 * q2))
        iterate = 0
        for j in range(q2):
            for i in range(p2):
                perturbation = np.zeros((p2, q2))
                perturbation[i, j] = epsilon
                nnet_minus.weights[w] = nnet_minus.weights[w] - perturbation
                nnet_plus.weights[w] = nnet_plus.weights[w] + perturbation

                nnet_minus.forward(training)
                nnet_minus.backward(training)

                nnet_plus.forward(training)
                nnet_plus.backward(training)

                delta = (nnet_plus.gradients[u] - nnet_minus.gradients[u]) / (2 * epsilon)
                Hessian_column = base.to_vector(delta)
                Hessian[:, iterate] = Hessian_column[:, 0]
                iterate = iterate + 1

                '''Reset weights'''
                nnet_minus.weights[w] = nnet_minus.weights[w] + perturbation
                nnet_plus.weights[w] = nnet_plus.weights[w] - perturbation

        Hessians.append(Hessian)

H = Hessians
hessian_FDM = np.bmat([[H[0], H[1], H[2]],
                        [H[3], H[4], H[5]],
                        [H[6], H[7], H[8]]])


'''Plot the Hessian Matrices and the errors'''
hessian_SOA = nnet.hessian_matrix
error_FDM = hessian_FDM - hessian_SOA

# Calculate bounds for color scaling
min_full = np.abs(np.min(hessian_SOA))
max_full = np.abs(np.max(hessian_SOA))
bound_full = np.max((min_full, max_full))

min_second = np.abs(np.min(hessian_FDM))
max_second = np.abs(np.max(hessian_FDM))
bound_second = np.max((min_second, max_second))

min_difference = np.abs(np.min(error_FDM))
max_difference = np.abs(np.max(error_FDM))
bound_difference = np.max((min_difference, max_difference))


plt.plot(nnet.costs)
plt.title('Cost Function Convergence')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.show()

# Create the plots
figure, axes = plt.subplots(1, 3, figsize=(18, 6))

im1 = axes[0].imshow(hessian_SOA, cmap='seismic', vmin=-bound_full, vmax=bound_full)
axes[0].set_xlabel('Weight Parameter ID (23 parameters)')
axes[0].set_ylabel('Weight Parameter ID (23 parameters)')
axes[0].set_title('Hessian Matrix - SOA')

im2 = axes[1].imshow(hessian_FDM, cmap='seismic', vmin=-bound_second, vmax=bound_second)
axes[1].set_xlabel('Weight Parameter ID (23 parameters)')
axes[1].set_ylabel('Weight Parameter ID (23 parameters)')
axes[1].set_title('Hessian Matrix - FDM')

im3 = axes[2].imshow(error_FDM, cmap='seismic', vmin=-bound_difference, vmax=bound_difference)
axes[2].set_xlabel('Weight Parameter ID (23 parameters)')
axes[2].set_ylabel('Weight Parameter ID (23 parameters)')
axes[2].set_title('FDM Error')

cbar1 = figure.colorbar(im1, ax=axes[0], shrink=0.6)
cbar2 = figure.colorbar(im2, ax=axes[1], shrink=0.6)
cbar3 = figure.colorbar(im3, ax=axes[2], shrink=0.6)

plt.tight_layout()
plt.show()

nnet.backward_hyperparameter_derivative(training)
print(nnet.dSW.shape)