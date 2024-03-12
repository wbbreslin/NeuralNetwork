import numpy as np
from NNET import neural_network
from Data import data
import ActivationFunctions as af
import CostFunctions as cf
import copy
import Base as base
import matplotlib.pyplot as plt

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
#np.random.seed(333)
training = data(x, y)
n = x.shape[0]
theta = 0.001/n
nnet = neural_network(layers=[2, 2, 3, 2],
                      activation_functions=[af.sigmoid,
                                            af.sigmoid,
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

nnet.train(training, max_iterations=itr1, step_size=step1)
nnet.train(training, max_iterations=itr2, step_size=step2)

'''Make copy of trained network to modify later'''
nnet_trained = copy.deepcopy(nnet)


'''Compute Hessian using SOA Model'''
nnet.forward(training)
nnet.backward(training)
nnet.backward_hyperparameter_derivative(training)
nnet.compute_hessian()

'''FDM approximate for W gradient on Validation Function'''
validation = data(x_validation,y_validation)

#Set up gradient matrices
#g0 = np.zeros(nnet.weights[0].shape)
#g1 = np.zeros(nnet.weights[1].shape)
#g2 = np.zeros(nnet.weights[2].shape)
#gradients = [g0, g1, g2]
gradients = [np.zeros(w.shape) for w in nnet.weights]

nnet_FDM = copy.deepcopy(nnet)
nnet_plus = copy.deepcopy(nnet_FDM)
nnet_minus = copy.deepcopy(nnet_FDM)

epsilon = 10**-6

for w in range(len(nnet.weights)):
    for row in range(nnet.weights[w].shape[0]):
        for col in range(nnet.weights[w].shape[1]):
            # Create separate copies of the weights for this iteration
            nnet_plus = copy.deepcopy(nnet_FDM)
            nnet_minus = copy.deepcopy(nnet_FDM)

            lo = nnet_minus.weights[w][row, col] - epsilon
            hi = nnet_plus.weights[w][row, col] + epsilon
            nnet_minus.weights[w][row, col] = lo
            nnet_plus.weights[w][row, col] = hi

            # Compute the network outputs with the updated weights
            nnet_plus.forward(validation)
            nnet_minus.forward(validation)
            #y_plus = data(x=None,y=nnet_plus.states[-1])
            #y_minus = data(x=None,y=nnet_minus.states[-1])


            # Compute cost for plus and minus weights
            nnet_minus.track_cost(validation)
            nnet_plus.track_cost(validation)
            cost_minus = nnet_minus.costs[-1]
            cost_plus = nnet_plus.costs[-1]

            # Compute the gradient using finite differences
            delta = (cost_plus - cost_minus) / (2 * epsilon)

            gradients[w][row, col] = delta

FDM_gradient = [base.to_vector(g) for g in gradients]
FDM_gradient = np.vstack(FDM_gradient)

nnet_q = copy.deepcopy(nnet_trained)
nnet_q.forward(validation)
nnet_q.backward(validation)
nnet_q.compute_gradient()

eta2 = np.linalg.solve(nnet.hessian_matrix, nnet_q.gradient_vector)


print(FDM_gradient-nnet_q.gradient_vector)


'''FDM for dJdSW'''
params = np.sum([w.size for w in nnet.weights])
dJdSW = np.zeros((10,params))
for i in range(10):
    training_FDM_plus = copy.deepcopy(training)
    training_FDM_minus = copy.deepcopy(training)
    training_FDM_plus.s = np.ones((10,1))
    training_FDM_minus.s = np.ones((10,1))
    training_FDM_plus.s[i] = training_FDM_plus.s[i]+epsilon
    training_FDM_minus.s[i] = training_FDM_minus.s[i]-epsilon
    nnet_plus = copy.deepcopy(nnet_trained)
    nnet_minus = copy.deepcopy(nnet_trained)
    #nnet_plus = copy.deepcopy(nnet_untrained)
    #nnet_minus = copy.deepcopy(nnet_untrained)
    #nnet_plus.train(training_FDM_plus, max_iterations=itr1, step_size=0.25)
    #nnet_plus.train(training_FDM_plus, max_iterations=itr2, step_size=0.05)
    #nnet_minus.train(training_FDM_minus, max_iterations=itr1, step_size=0.25)
    #nnet_minus.train(training_FDM_minus, max_iterations=itr2, step_size=0.05)
    nnet_plus.forward(training_FDM_plus)
    nnet_minus.forward(training_FDM_minus)
    nnet_plus.backward(training_FDM_plus)
    nnet_minus.backward(training_FDM_minus)
    nnet_plus.compute_gradient()
    nnet_minus.compute_gradient()
    dSW_component = ((nnet_plus.gradient_vector - nnet_minus.gradient_vector)/ (2 * epsilon)).T
    dJdSW[i,:] = dSW_component


plt.plot(nnet.costs)
plt.show()

'''Plot the Hessian Matrices and the errors'''
#matrix = nnet.dSW
matrix = dJdSW
hessian_inv = np.linalg.inv(nnet.hessian_matrix)
product = matrix @ hessian_inv
hessian = nnet.hessian_matrix

# Calculate bounds for color scaling
min_full = np.abs(np.min(matrix))
max_full = np.abs(np.max(matrix))
bound_full = np.max((min_full, max_full))

min_inv = np.abs(np.min(hessian_inv))
max_inv = np.abs(np.max(hessian_inv))
bound_inv = np.max((min_inv, max_inv))

min_prod = np.abs(np.min(product))
max_prod = np.abs(np.max(product))
bound_prod = np.max((min_prod, max_prod))

min_hess = np.abs(np.min(hessian))
max_hess = np.abs(np.max(hessian))
bound_hess = np.max((min_hess, max_hess))

# Create the plots
figure, axes = plt.subplots(1, 3, figsize=(18, 12))

im1 = axes[0].imshow(matrix, cmap='seismic', vmin=-bound_full, vmax=bound_full)
axes[0].set_title('SW Derivative of Cost')
cbar1 = figure.colorbar(im1, ax=axes[0], orientation='horizontal')


im2 = axes[1].imshow(hessian_inv, cmap='seismic', vmin=-bound_inv, vmax=bound_inv)
axes[1].set_title('Inverse Hessian of Cost')
cbar2 = figure.colorbar(im2, ax=axes[1], orientation='horizontal')
axes[1].set_aspect(17/17)

im3 = axes[2].imshow(product, cmap='seismic', vmin=-bound_prod, vmax=bound_prod)
axes[2].set_title('Product: SW(J) x Inverse Hessian')
cbar3 = figure.colorbar(im3, ax=axes[2], orientation='horizontal')


#plt.tight_layout()
plt.show()


'''Eigenvalue plot'''
eigenvalues, _ = np.linalg.eig(nnet.hessian_matrix)

# Plot the eigenvalues on the real axis
plt.scatter(np.arange(len(eigenvalues)), eigenvalues, marker='o', color='blue')

# Highlight positive and negative eigenvalues with different colors
positive_eigenvalues = eigenvalues[eigenvalues > 0]
negative_eigenvalues = eigenvalues[eigenvalues < 0]

plt.scatter(np.where(eigenvalues > 0)[0], positive_eigenvalues, marker='o', color='blue', label='Positive Eigenvalues')
plt.scatter(np.where(eigenvalues < 0)[0], negative_eigenvalues, marker='o', color='red', label='Negative Eigenvalues')

plt.title("Eigenvalues of the Hessian")
plt.xlabel("Eigenvalue Index")
plt.ylabel("Eigenvalue")
plt.axhline(0, color='black', linewidth=0.5)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()
