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

df = data(x_predictors,y_outcomes)
training, validation = df.test_train_split(train_percent=0.4)
nnet = neural_network(layers=[4,8,8,2],
                         activation_functions = [af.sigmoid,
                                                 af.sigmoid,
                                                 af.sigmoid],
                         cost_function=cf.mean_squared_error)

nnet.randomize_weights()

nnet.train(training, max_iterations = 2000, step_size=0.05)

nnet.predict(validation)
print(np.round(nnet.predictions)-validation.y)


# Sensitivity Analysis
print('Beginning Sensitivity Analysis')
nnet.compute_hessian()
print('Hessian computed')
nnet.compute_gradient()
print('Gradient computed')
eta = np.linalg.inv(nnet.hessian_matrix) @ nnet.gradient_vector
print('Eta computed')

nnet.backward_hyperparameter_derivative(training)
n = training.x.shape[0]
array = list(range(n**2))
index = [array[i] for i in range(len(array)) if i % (n+1) == 0]

total_dJ = np.hstack(nnet.dJ)
total_dJ = total_dJ[index,:]

sensitivity = total_dJ @ eta
print(sensitivity)


plt.plot(nnet.costs)
plt.show()

'''
Things to do:
* OSE, and validate FSO calculations
* Solve for eta using hessian vector products and solving linear system
'''