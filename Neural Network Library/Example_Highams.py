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

# Sigmoid is working, softmax output is not
training = data(x,y)
nnet.forward(training)
nnet.backward(training)
nnet.train(training, max_iterations =3000, step_size=0.25) #3000
plt.plot(nnet.costs)
plt.show()

vectors = nnet.gradients
nnet.soa_forward(vectors) #This is working, but need to validate results...
print([h.shape for h in nnet.activation_hessian_matrices])

A = nnet.Av_Tensor(2)
v = [base.to_vector(g) for g in nnet.gradients]
D = nnet.Dv_Tensor(vectors=v, i=2)
print(D.shape)
print(D)
C = nnet.Cv_Tensor(2)
print(C.shape)