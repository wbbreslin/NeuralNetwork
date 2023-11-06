import numpy as np
import Base as base
import FirstOrderModel as fom
import TrainingAlgorithms as train
import copy
from scipy.sparse import block_diag
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
nnet = {'Predictors': x_predictors,
        'Outcomes': y_outcomes,
        'Weights': weights}

'''Exact gradient calculation'''
nnet = train.gradient_descent(nnet,max_iterations=4000)
nnet_plus = copy.deepcopy(nnet)
nnet_minus = copy.deepcopy(nnet)

epsilon = 10**-6

Hessians = []
for w in range(len(nnet['Weights'])):
    shape = nnet['Weights'][w].shape
    p = shape[0]
    q = shape[1]
    Hessian = np.zeros((p*q,p*q))
    iterate = 0
    for j in range(q):
        for i in range(p):
            perturbation = np.zeros(shape)
            perturbation[i,j] = epsilon
            nnet_minus['Weights'][w] = nnet_minus['Weights'][w] - perturbation
            nnet_plus['Weights'][w] = nnet_plus['Weights'][w] + perturbation

            nnet_minus = fom.forward_pass(nnet_minus)
            nnet_minus = fom.backward_pass(nnet_minus)

            nnet_plus = fom.forward_pass(nnet_plus)
            nnet_plus = fom.backward_pass(nnet_plus)

            delta = (nnet_plus['Gradients'][w] - nnet_minus['Gradients'][w]) / (2 * epsilon)
            Hessian_column, dim = base.to_vector(delta)
            Hessian[:,iterate] = Hessian_column[:,0]
            iterate = iterate + 1

            '''Reset weights'''
            nnet_minus['Weights'][w] = nnet_minus['Weights'][w] + perturbation
            nnet_plus['Weights'][w] = nnet_plus['Weights'][w] - perturbation

    Hessians.append(Hessian)

#print(np.round(Hessians[0],4))
#print(np.round(Hessians[1],4))
#print(np.round(Hessians[2],4))
A = Hessians[0]
B = Hessians[1]
C = Hessians[2]

full_hessian = block_diag((A,B,C)).toarray()
full_hessian[full_hessian==0] = np.nan


"""Plot the Hessian"""
#plt.figure(figsize=(23, 23))  # Set the figure size as needed
vmin, vmax = full_hessian.min(), full_hessian.max()

# Use imshow to create the heatmap
plt.imshow(full_hessian, cmap='viridis')  # You can choose a different colormap if needed

# Add colorbar for reference
plt.colorbar()

# Optionally, add labels and title
plt.xlabel('Weight Parameter Index (out of 23)')
plt.ylabel('Weight Parameter Index (out of 23)')
plt.title('Hessian Matrix (4000 Iterations)')

# Show the heatmap
plt.show()