import numpy as np
from NNET import neural_network
from Data import data
import ActivationFunctions as af
import CostFunctions as cf
import copy
import Base as base
import matplotlib.pyplot as plt

'''Data'''
x = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],
              [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]]).T

y = np.array([[1,1,1,1,1,0,0,0,0,0],
              [0,0,0,0,0,1,1,1,1,1]]).T

x_validation = np.array([[0.7,0.2,0.6,0.9],
                         [0.9,0.7,0.1,0.8]]).T

y_validation = np.array([[1,1,0,0],
                         [0,0,1,1]]).T

'''
x_validation = np.array([[0.5,0.1,0.2,0.7,0.2,0.6,0.9,0.8,0.6,0.8],
                         [0.1,0.9,0.3,0.9,0.7,0.1,0.8,0.4,0.6,0.1]]).T
                         
y_validation = np.array([[1,1,1,1,1,0,0,0,0,0],
                         [0,0,0,0,0,1,1,1,1,1]]).T
'''



'''Define the model'''
#np.random.seed(333)
training = data(x,y)
validation = data(x_validation,y_validation)
nnet = neural_network(layers=[2,2,3,2],
                         activation_functions = [af.sigmoid,
                                                 af.sigmoid,
                                                 af.sigmoid],
                         cost_function=cf.half_SSE)
nnet.randomize_weights()

'''Make copy of untrained network for OSE'''
nnet_OSE_ref = copy.deepcopy(nnet)

'''Train the network'''
itr1 = 6000 ; step1 = 0.25
itr2 = 4000; step2 = 0.05
nnet.train(training, max_iterations = itr1, step_size=0.25)
nnet.train(training, max_iterations = itr2, step_size=0.05)

'''Make copy of trained network for validation functional'''
nnet_copy = copy.deepcopy(nnet)
nnet_FSO_copy = copy.deepcopy(nnet)
nnet_FDM = copy.deepcopy(nnet)

nnet.compute_gradient()
nnet.compute_hessian()
nnet.backward_hyperparameter_derivative(training)

'''FDM Hessian'''
nnet_plus = copy.deepcopy(nnet_FDM)
nnet_minus = copy.deepcopy(nnet_FDM)
epsilon = 10**-6

Hessians = []
for u in range(len(nnet.weights)):
    for w in range(len(nnet.weights)):
        shape_num = nnet.weights[u].shape
        shape_den = nnet.weights[w].shape
        p1 = shape_num[0]
        q1 = shape_num[1]
        p2 = shape_den[0]
        q2 = shape_den[1]
        Hessian = np.zeros((p1*q1,p2*q2))
        iterate = 0
        for j in range(q2):
            for i in range(p2):
                perturbation = np.zeros((p2,q2))
                perturbation[i,j] = epsilon
                nnet_minus.weights[w] = nnet_minus.weights[w] - perturbation
                nnet_plus.weights[w] = nnet_plus.weights[w] + perturbation

                nnet_minus.forward(training)
                nnet_minus.backward(training)

                nnet_plus.forward(training)
                nnet_plus.backward(training)

                delta = (nnet_plus.gradients[u] - nnet_minus.gradients[u]) / (2 * epsilon)
                Hessian_column = base.to_vector(delta)
                Hessian[:,iterate] = Hessian_column[:,0]
                iterate = iterate + 1

                '''Reset weights'''
                nnet_minus.weights[w] = nnet_minus.weights[w] + perturbation
                nnet_plus.weights[w] = nnet_plus.weights[w] - perturbation

        Hessians.append(Hessian)

H = Hessians
full_hessian = np.bmat([[H[0], H[1], H[2]],
                       [H[3], H[4], H[5]],
                       [H[6], H[7], H[8]]])

print(full_hessian-nnet.hessian_matrix)


second_hessian = nnet.hessian_matrix
difference_hessian = full_hessian - second_hessian

# Calculate bounds for color scaling
min_full = np.abs(np.min(full_hessian))
max_full = np.abs(np.max(full_hessian))
bound_full = np.max((min_full, max_full))

min_second = np.abs(np.min(second_hessian))
max_second = np.abs(np.max(second_hessian))
bound_second = np.max((min_second, max_second))

min_difference = np.abs(np.min(difference_hessian))
max_difference = np.abs(np.max(difference_hessian))
bound_difference = np.max((min_difference, max_difference))

# Create subplots with 1 row and 3 columns
figure, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot the first heatmap (full_hessian) on the left subplot
im1 = axes[0].imshow(full_hessian, cmap='seismic', vmin=-bound_full, vmax=bound_full)
axes[0].set_xlabel('Weight Parameter ID (23 parameters)')
axes[0].set_ylabel('Weight Parameter ID (23 parameters)')
axes[0].set_title('Full Hessian')

# Plot the second heatmap (second_hessian) in the middle subplot
im2 = axes[1].imshow(second_hessian, cmap='seismic', vmin=-bound_second, vmax=bound_second)
axes[1].set_xlabel('Weight Parameter ID (23 parameters)')
axes[1].set_ylabel('Weight Parameter ID (23 parameters)')
axes[1].set_title('Second Hessian')

# Plot the third heatmap (difference_hessian) on the right subplot
im3 = axes[2].imshow(difference_hessian, cmap='seismic', vmin=-bound_difference, vmax=bound_difference)
axes[2].set_xlabel('Weight Parameter ID (23 parameters)')
axes[2].set_ylabel('Weight Parameter ID (23 parameters)')
axes[2].set_title('Difference Hessian')

# Add colorbars to each subplot
cbar1 = figure.colorbar(im1, ax=axes[0])
cbar2 = figure.colorbar(im2, ax=axes[1])
cbar3 = figure.colorbar(im3, ax=axes[2])


# Show the plot
plt.show()