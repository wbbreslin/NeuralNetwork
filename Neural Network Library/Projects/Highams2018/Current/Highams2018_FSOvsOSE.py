import numpy as np
from NNET import neural_network
from Data import data
import ActivationFunctions as af
import CostFunctions as cf
import copy
import matplotlib.pyplot as plt

'''Data'''
x = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],
              [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]]).T

y = np.array([[1,1,1,1,1,0,0,0,0,0],
              [0,0,0,0,0,1,1,1,1,1]]).T

x_validation = np.array([[0.7,0.2,0.6,0.9],
                         [0.9,0.7,0.1,0.8]]).T

y_validation = np.array([[1,1,0,0],
                         [0,0,1,1]]).T

'''
x_validation = np.array([[0.5,0.1,0.2,0.7,0.2,0.6,0.9,0.8,0.6,0.8],
                         [0.1,0.9,0.3,0.9,0.7,0.1,0.8,0.4,0.6,0.1]]).T
                         
y_validation = np.array([[1,1,1,1,1,0,0,0,0,0],
                         [0,0,0,0,0,1,1,1,1,1]]).T
'''

'''Define the model'''
#np.random.seed(333)
training = data(x,y)
validation = data(x_validation,y_validation)
nnet = neural_network(layers=[2,2,3,2],
                         activation_functions = [af.sigmoid,
                                                 af.sigmoid,
                                                 af.sigmoid],
                         cost_function=cf.half_SSE)

'''Duplicate untrained model for OSE'''
nnet_OSE_init = copy.deepcopy(nnet)

'''Train to optimality'''
itr1 = 6000 ; step1 = 0.25
itr2 = 4000; step2 = 0.05
nnet.train(training, max_iterations = itr1, step_size=0.25)
nnet.train(training, max_iterations = itr2, step_size=0.05)

'''Duplicate trained model for FSO'''
nnet_FSO_J = copy.deepcopy(nnet)
nnet_FSO_q = copy.deepcopy(nnet)

'''FSO Sensitivity Analysis'''
nnet_FSO_J.forward(training)
nnet_FSO_J.backward(training)
nnet_FSO_J.backward_hyperparameter_derivative(training)
nnet_FSO_J.compute_gradient()
nnet_FSO_J.compute_hessian()

nnet_FSO_q.forward(validation)
nnet_FSO_q.backward(validation)
nnet_FSO_q.track_cost(validation)
nnet_FSO_q.compute_gradient()
print(nnet_FSO_q.costs[-1])
eta = np.linalg.inv(nnet_FSO_J.hessian_matrix) @ nnet_FSO_q.gradient_vector
forecast_gradient = -nnet_FSO_J.dJ @ eta
sensitivity = -forecast_gradient



'''OSE Sensitivity Analysis'''
#Training - Compute Unmodified Validation Cost
nnet_OSE = copy.deepcopy(nnet_OSE_init)
nnet_OSE.train(training, max_iterations = itr1, step_size=0.25)
nnet_OSE.train(training, max_iterations = itr2, step_size=0.05)
nnet_OSE_validation = copy.deepcopy(nnet_OSE)
nnet_OSE_validation.forward(validation)
nnet_OSE_validation.backward(validation)
nnet_OSE_validation.track_cost(validation)
unmodified_cost = nnet_OSE_validation.costs[-1]

cost_impact = []
for i in range(10):
    nnet_OSE = copy.deepcopy(nnet_OSE_init)
    x_OSE = np.delete(training.x,i,axis=0)
    y_OSE = np.delete(training.y,i,axis=0)
    training_OSE = data(x_OSE,y_OSE)
    nnet_OSE.train(training_OSE, max_iterations=itr1, step_size=0.25)
    nnet_OSE.train(training_OSE, max_iterations=itr2, step_size=0.05)
    nnet_OSE_validation = copy.deepcopy(nnet_OSE)
    nnet_OSE_validation.forward(validation)
    nnet_OSE_validation.backward(validation)
    nnet_OSE_validation.track_cost(validation)
    new_cost = nnet_OSE_validation.costs[-1]
    delta = new_cost-unmodified_cost
    cost_impact.append(delta)

'''Plots'''
figure, axis = plt.subplots(2, 2)
axis[0,0].plot(nnet.costs)

#region = region_plot(nnet,training)
#axis[0,1].plot(region)

cost_impact = np.array(cost_impact).reshape(-1,1)
axis[1,0].plot(sensitivity)
axis[1,0].plot(cost_impact)
plt.show()
