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
theta = 0.018/n
nnet = neural_network(layers=[2, 2, 3, 2],
                      activation_functions=[af.sigmoid,
                                            af.sigmoid,
                                            af.sigmoid],
                      cost_function=cf.half_SSE,
                      regularization=theta)

''' Copy untrained network for FDM of dJdSW'''
nnet_untrained = copy.deepcopy(nnet)

'''Train the network'''
itr1 = 6000
step1 = 0.25
itr2 = 4000
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