import FirstOrderModel as fom
import SecondOrderModel as som
import TrainingAlgorithms as train
import numpy as np
import Base as base

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
                           neurons = [2,2,3,2],
                           activations = [base.sigmoid,
                                          base.sigmoid,
                                          base.sigmoid])

def prune_network(nnet):
    weights = nnet['Weights']
    pruning_matrices = [np.ones(x.shape) for x in weights]
    nnet = train.pruned_gradient_descent(nnet, pruning_matrices, step_size=0.25, max_iterations=4000)

    return(nnet)

nnet2 = prune_network(nnet)
