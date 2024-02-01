import numpy as np
import Base as base
import matplotlib.pyplot as plt
import NetworkPruning as prune

"""The data set of predictor variables"""
x_predictors = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],
                        [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]]).T

"""The data set of outcome variables"""
y_outcomes = np.array([[1,1,1,1,1,0,0,0,0,0],
                      [0,0,0,0,0,1,1,1,1,1]]).T

"""Define the neural network structure"""
np.random.seed(100)
nnet = base.create_network(x_predictors,
                           y_outcomes,
                           neurons = [2,4,4,2],
                           activations = [base.sigmoid,
                                          base.sigmoid,
                                          base.sigmoid])

nnet = prune.iterative_prune(nnet,itr=29, remove=1)

plt.plot(np.log(nnet['Cost']))
plt.plot(nnet['Cost'])
plt.show()