import numpy as np
from NNET import neural_network
from Data import data
import ActivationFunctions as af
import CostFunctions as cf
import copy
import matplotlib.pyplot as plt

"""The data set of predictor variables"""
x = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],
              [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]]).T


"""The data set of outcome variables"""
y = np.array([[1,1,1,1,1,0,0,0,0,0],
              [0,0,0,0,0,1,1,1,1,1]]).T

np.random.seed(100)

df = data(x,y)
training, validation = df.test_train_split(train_percent=0.6)
nnet = neural_network(layers=[2,2,3,2],
                         activation_functions = [af.sigmoid,
                                                 af.sigmoid,
                                                 af.sigmoid],
                         cost_function=cf.mean_squared_error)

nnet.randomize_weights()

# Duplicating Untrained NNET for Observing System Experiment (same starting weights)
nnet_OSE_base = copy.deepcopy(nnet)

nnet.train(training, max_iterations = 4000, step_size=0.25)
plt.plot(nnet.costs,label=f'Reference')

print(training.x.shape)
print(training.y.shape)

# Sensitivity Analysis
nnet.compute_hessian()
nnet.compute_gradient()
eta = np.linalg.inv(nnet.hessian_matrix) @ nnet.gradient_vector

nnet.backward_hyperparameter_derivative(training)
n = training.x.shape[0]
array = list(range(n**2))
index = [array[i] for i in range(len(array)) if i % (n+1) == 0]

total_dJ = np.hstack(nnet.dJ)
total_dJ = total_dJ[index,:]

sensitivity = total_dJ @ eta
print(sensitivity)
print(training.x)

OSE = []
unmodified_cost = nnet.costs[-1]
# Observing System Experiment
for i in range(10):
    nnet_OSE = copy.deepcopy(nnet_OSE_base)
    x_OSE = np.delete(x,i,axis=0)
    y_OSE = np.delete(y,i,axis=0)
    df_OSE = data(x_OSE,y_OSE)
    training, validation = df_OSE.test_train_split(train_percent=1)
    nnet_OSE.train(df_OSE, max_iterations=4000, step_size=0.25)
    new_cost = nnet_OSE.costs[-1]
    delta = new_cost - unmodified_cost
    OSE.append(delta)
    plt.plot(nnet_OSE.costs,label=f'Iteration {i + 1}')


print(OSE)
plt.legend()
plt.show()

'''
Need to do test train split - fix this
Then measure error on validation functional, not cost
'''