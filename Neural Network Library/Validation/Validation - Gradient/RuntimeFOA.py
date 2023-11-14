import numpy as np
import FirstOrderModel as fom
import TrainingAlgorithms as train
import timeit

"""
Description:
First-order adjoint evaluation of the gradient for a neural network

Dataset: 
SIAM 2019
"""

x_predictors = np.array([[0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7],
                        [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]]).T

y_outcomes = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]).T

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

weights = [w0,w1,w2]
neurons = np.array([2,2,3,2])
activations = ["sigmoid","sigmoid","sigmoid","sigmoid"]

nnet = {'Predictors': x_predictors,
        'Outcomes': y_outcomes,
        'Weights': weights,
        'Neurons': neurons}

start = timeit.default_timer()

for i in range(4000):
    nnet = fom.forward_pass(nnet)
    nnet = fom.backward_pass(nnet)

stop = timeit.default_timer()
time = stop-start

print(time)