import numpy as np
import Base as base
import FirstOrderModel as fom
import TrainingAlgorithms as train
import copy

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
training_itr = 4000
nnet = train.gradient_descent(nnet,max_iterations=training_itr)
nnet_plus = copy.deepcopy(nnet)
nnet_minus = copy.deepcopy(nnet)

epsilon = 10**-6

Hessians = []
for u in range(len(nnet['Weights'])):
    for w in range(len(nnet['Weights'])):
        shape_num = nnet['Weights'][u].shape
        shape_den = nnet['Weights'][w].shape
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
                nnet_minus['Weights'][w] = nnet_minus['Weights'][w] - perturbation
                nnet_plus['Weights'][w] = nnet_plus['Weights'][w] + perturbation

                nnet_minus = fom.forward_pass(nnet_minus)
                nnet_minus = fom.backward_pass(nnet_minus)

                nnet_plus = fom.forward_pass(nnet_plus)
                nnet_plus = fom.backward_pass(nnet_plus)

                delta = (nnet_plus['Gradients'][u] - nnet_minus['Gradients'][u]) / (2 * epsilon)
                Hessian_column, dim = base.to_vector(delta)
                Hessian[:,iterate] = Hessian_column[:,0]
                iterate = iterate + 1

                '''Reset weights'''
                nnet_minus['Weights'][w] = nnet_minus['Weights'][w] + perturbation
                nnet_plus['Weights'][w] = nnet_plus['Weights'][w] - perturbation

        Hessians.append(Hessian)

H = Hessians
full_hessian = np.bmat([[H[0], H[1], H[2]],
                       [H[3], H[4], H[5]],
                       [H[6], H[7], H[8]]])

print(full_hessian - full_hessian.T)