import numpy as np
from NNET import neural_network
from Data import data
import ActivationFunctions as af
import CostFunctions as cf
import copy
import Base as base


'''Data'''
x = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],
              [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]]).T

x_validation = np.array([[0.7,0.2,0.6,0.9],
                         [0.9,0.7,0.1,0.8]]).T

x_validation = np.array([[0.5,0.1,0.2,0.7,0.2,0.6,0.9,0.8,0.6,0.8],
                         [0.1,0.9,0.3,0.9,0.7,0.1,0.8,0.4,0.6,0.1]]).T

y = np.array([[1,1,1,1,1,0,0,0,0,0],
              [0,0,0,0,0,1,1,1,1,1]]).T

y_validation = np.array([[1,1,0,0],
                         [0,0,1,1]]).T

y_validation = np.array([[1,1,1,1,1,0,0,0,0,0],
                         [0,0,0,0,0,1,1,1,1,1]]).T


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

nnet.compute_gradient()
nnet.compute_hessian()
nnet.backward_hyperparameter_derivative(training)