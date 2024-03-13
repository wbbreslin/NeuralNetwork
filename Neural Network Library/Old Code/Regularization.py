import numpy as np
from NNET import neural_network
from Data import data
import ActivationFunctions as af
import CostFunctions as cf
import matplotlib.pyplot as plt
import copy

'''Data'''
x = np.array([[0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7],
              [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]]).T

y = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]).T

x_validation = np.array([[0.7,0.2,0.6,0.9],
                         [0.9,0.7,0.1,0.8]]).T

y_validation = np.array([[1,1,0,0],
                         [0,0,1,1]]).T

'''Define the model'''
np.random.seed(333)
training = data(x, y)

n = x.shape[0]
theta = 0.001/n
nnet = neural_network(layers=[2, 3, 2],
                      activation_functions=[af.linear,
                                            af.sigmoid],
                      cost_function=cf.half_SSE,
                      regularization=theta)

''' Copy untrained network for FDM of dJdSW'''
nnet_untrained = copy.deepcopy(nnet)

'''Train the network'''
itr1 = 5000
step1 = 0.25
itr2 = 5000
step2 = 0.05

nnet.train(training, max_iterations=itr1, step_size=0.25)
nnet.train(training, max_iterations=itr2, step_size=0.05)

'''Make copy of trained network to modify later'''
nnet_trained = copy.deepcopy(nnet)

plt.plot(nnet.costs)
plt.show()

print(nnet.weights[0])
print(np.linalg.norm(nnet.augmented_weights[0]))

nnet.predict(training)
print(np.round(training.predictions,2))

nnet_FSO_J = copy.deepcopy(nnet)
nnet_FSO_q = copy.deepcopy(nnet)

nnet.compute_hessian()

matrix = nnet.hessian_matrix
eigenvalues, _ = np.linalg.eig(nnet.hessian_matrix)
positive_eigenvalues = eigenvalues[eigenvalues > 0]
negative_eigenvalues = eigenvalues[eigenvalues < 0]

# Calculate bounds for color scaling
min_full = np.abs(np.min(matrix))
max_full = np.abs(np.max(matrix))
bound_full = np.max((min_full, max_full))
figure, axes = plt.subplots(1, 2, figsize=(18, 12),gridspec_kw={'width_ratios': [1, 1]})

im1 = axes[0].imshow(matrix, cmap='seismic', vmin=-bound_full, vmax=bound_full)
axes[0].set_title('Hessian')
cbar1 = figure.colorbar(im1, ax=axes[0], shrink=0.6)
axes[0].set_xlabel('Weight Parameter ID (17 parameters)')

axes[1].scatter(np.arange(len(eigenvalues)), eigenvalues, marker='o', color='blue')
axes[1].scatter(np.where(eigenvalues > 0)[0], positive_eigenvalues, marker='o', color='blue', label='Positive Eigenvalues')
axes[1].scatter(np.where(eigenvalues < 0)[0], negative_eigenvalues, marker='o', color='red', label='Negative Eigenvalues')
axes[1].set_title('Eigenvalues')
axes[1].set_xlabel('Eigenvalue Index')
axes[1].set_ylabel('Eigenvalue')
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].grid(True, linestyle='--', alpha=0.7)
axes[1].set_box_aspect(0.9)
axes[1].legend()

plt.show()