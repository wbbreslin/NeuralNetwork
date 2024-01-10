import Base as base
from sklearn import datasets
import numpy as np

'''Import the data from sklearn'''
iris = datasets.load_iris()

'''Pull predictors and Outcomes from data'''
x_predictors = iris['data']
y = iris['target']

'''Eliminate the third category for binary classification'''
subset = y!=2
x_predictors = x_predictors[subset]
y = y[subset]

'''Label classes as 2d vectors'''
y_outcomes = np.zeros((len(y),2))
y_outcomes[y==0] = np.array([[1,0]])
y_outcomes[y==1] = np.array([[0,1]])

'''Define the neural network'''
np.random.seed(100)
nnet = base.create_network(x_predictors,
                           y_outcomes,
                           neurons = [4,8,8,2],
                           activations = ["sigmoid","sigmoid","sigmoid"])


