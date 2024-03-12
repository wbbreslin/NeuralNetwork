import numpy as np
from NNET import neural_network
from Data import data
import ActivationFunctions as af
import CostFunctions as cf
import matplotlib.pyplot as plt
import copy
import Base as base

'''Data'''
x = np.array([[0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7],
              [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]]).T

y = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]).T

x_validation = np.array([[0.7, 0.2, 0.6, 0.9],
                         [0.9, 0.7, 0.1, 0.8]]).T

y_validation = np.array([[1, 1, 0, 0],
                         [0, 0, 1, 1]]).T


'''Define the network structure'''
np.random.seed(333)
theta = 0
nnet = neural_network(layers=[2, 2, 3, 2],
                      activation_functions=[af.sigmoid,
                                            af.sigmoid,
                                            af.sigmoid],
                      cost_function=cf.half_SSE,
                      regularization=theta)
training = data(x, y)
validation = data(x_validation, y_validation)


'''Train the network'''
itr1 = 4000; step1 = 0.25
itr2 = 2000; step2 = 0.0001

nnet.train(training, max_iterations=itr1, step_size=step1)
nnet.train(training, max_iterations=itr2, step_size=step2)

'''Copy of trained network for validation'''
#nnet_validation = copy.deepcopy(nnet)

'''Plot Cost Function'''
plt.plot(nnet.costs)
plt.show()

'''Training Accuracy'''
nnet.predict(training)
training_incorrect = np.sum(np.sum((np.round(training.predictions) - training.y)**2, axis=1))
training_correct = training.x.shape[0] - training_incorrect
training_accuracy = 100*training_correct/training.x.shape[0]
print("Training Accuracy:", training_accuracy,"%")

'''Validation Accuracy'''
nnet.predict(validation)
validation_incorrect = np.sum(np.sum((np.round(validation.predictions) - validation.y)**2, axis=1))
validation_correct = validation.x.shape[0] - validation_incorrect
validation_accuracy = 100*validation_correct/validation.x.shape[0]
print("Validation Accuracy:", validation_accuracy,"%")

v = [base.to_vector(g) for g in nnet.gradients]
nnet.Hvp(v)