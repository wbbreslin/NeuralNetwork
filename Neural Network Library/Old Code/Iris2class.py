import Base as base
from sklearn import datasets
import numpy as np
from NNET import neural_network
from Data import data
import ActivationFunctions as af
import CostFunctions as cf
import matplotlib.pyplot as plt

'''Import the data from sklearn'''
iris = datasets.load_iris()

'''Pull predictors and Outcomes from data'''
x_predictors = iris['data']
y = iris['target']

'''Eliminate the third category for binary classification'''
subset = y!=0
x_predictors = x_predictors[subset]
y = y[subset]

'''Label classes as 2d vectors'''
y_outcomes = np.zeros((len(y),2))
y_outcomes[y==1] = np.array([[1,0]])
y_outcomes[y==2] = np.array([[0,1]])

'''Test-train split (80-20)'''
rows = x_predictors.shape[0]
indices = base.generate_random_indices(rows,random_seed=100)
split = int(np.round(rows*0.8))
train_indices = indices[0:split]
test_indices = indices[split:]
x_train = x_predictors[train_indices]
x_test = x_predictors[test_indices]
y_train = y_outcomes[train_indices]
y_test = y_outcomes[test_indices]

'''Define the neural network'''
np.random.seed(100)

training = data(x_train,y_train)
validation = data(x_test, y_test)
nnet = neural_network(layers=[4, 8, 2],
                      activation_functions=[af.relu,
                                            af.sigmoid],
                      cost_function=cf.half_SSE)

nnet.weights = [0*w for w in nnet.weights]

step = 0.4
itr = 1000
nnet.train(training, max_iterations=itr, step_size=step)
nnet.predict(validation)
print(np.round(validation.predictions)-validation.y)

plt.plot(nnet.costs)
plt.show()

print(nnet.costs[-1])
