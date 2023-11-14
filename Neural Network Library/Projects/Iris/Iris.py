import Base as base
import TrainingAlgorithms as train
from sklearn import datasets
import numpy as np

'''Import the data from sklearn'''
iris = datasets.load_iris()

'''Pull predictors and Outcomes from data'''
x_predictors = iris['data']

'''Convert y categories to vectors'''
y = iris['target']
y_outcomes = np.zeros((len(y),3))
y_outcomes[y==0] = np.array([[1,0,0]])
y_outcomes[y==1] = np.array([[0,1,0]])
y_outcomes[y==2] = np.array([[0,0,1]])

"""Define the neural network structure"""
np.random.seed(100)
nnet = base.create_network(x_predictors,
                           y_outcomes,
                           neurons = [4,16,3],
                           activations = ["ReLU","Softmax"])

# runtime for 10**4 iterations is 902 seconds, about 15 minutes
nnet = train.gradient_descent(nnet, max_iterations=10**4)
base.store_nnet(nnet)