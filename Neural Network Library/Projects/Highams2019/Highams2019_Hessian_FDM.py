import Highams2019_Train_GradientDescent as data
import numpy as np
import Base as base
import FirstOrderModel as fom
import copy

nnet = data.nnet


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
                Hessian_column = base.to_vector(delta)
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