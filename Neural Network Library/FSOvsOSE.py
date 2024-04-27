import numpy as np
from NeuralNetwork import neural_network
from Data import data
import ActivationFunctions as af
import CostFunctions as cf
import copy
import matplotlib.pyplot as plt
import warnings
from scipy.stats import spearmanr
from matplotlib.cm import ScalarMappable

'''Data'''
x = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],
              [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]])

y = np.array([[1,1,1,1,1,0,0,0,0,0],
              [0,0,0,0,0,1,1,1,1,1]])


x_validation = np.array([[0.7,0.2,0.6,0.9],
                         [0.9,0.7,0.1,0.8]])

y_validation = np.array([[1,1,0,0],
                         [0,0,1,1]])



'''Define the model'''
#np.random.seed(333)
training = data(x,y)
validation = data(x_validation,y_validation)

n = x.shape[1]
nnet = neural_network(layers=[2,2,3,2],
                      activation_functions = [af.sigmoid,
                                              af.sigmoid,
                                              af.softmax],
                      cost_function=cf.half_SSE)

'''Pretraining'''
itr1 = 5000 ; step1 = 0.25 #6000 and 0.25
nnet.train(training, max_iterations=itr1, step_size=step1)

'''Duplicate pretrained model for OSE'''
nnet_OSE_init = copy.deepcopy(nnet)

'''Additional training'''
itr2 = 3000; step2 = 0.25
nnet.train(training, max_iterations=itr2, step_size=step2)

'''Duplicate trained model for FSO'''
nnet_FSO_J = copy.deepcopy(nnet)
nnet_FSO_q = copy.deepcopy(nnet)


'''FSO Sensitivity Analysis'''
nnet_FSO_J.forward(training)
nnet_FSO_J.backward(training)
nnet_FSO_J.backward_hyperparameter_derivative(training)
nnet_FSO_J.compute_hessian()

nnet_FSO_q.forward(validation)
nnet_FSO_q.backward(validation)
nnet_FSO_q.track_cost(validation)
nnet_FSO_q.compute_gradient()

eta = np.linalg.inv(nnet_FSO_J.hessian_matrix) @ nnet_FSO_q.gradient_vector
forecast_gradient = -nnet_FSO_J.dSW @ eta
sensitivity = -forecast_gradient


'''OSE Sensitivity Analysis'''
#Training - Compute Unmodified Validation Cost
nnet_OSE_validation = copy.deepcopy(nnet)
nnet_OSE_validation.forward(validation)
nnet_OSE_validation.track_cost(validation)
unmodified_cost = nnet_OSE_validation.costs[-1]
print("Original validation cost:", unmodified_cost)


cost_impact = []
perturbation = 0.001
for i in range(x.shape[1]):
    nnet_OSE = copy.deepcopy(nnet_OSE_init)
    training_OSE = copy.deepcopy(training)
    training_OSE.s = np.ones((x.shape[1],1))
    training_OSE.s[i] = 1-perturbation
    nnet_OSE.train(training_OSE, max_iterations=itr2, step_size=step2)
    nnet_OSE_validation = copy.deepcopy(nnet_OSE)
    nnet_OSE_validation.forward(validation)
    nnet_OSE_validation.track_cost(validation)
    new_cost = nnet_OSE_validation.costs[-1]
    delta = new_cost-unmodified_cost
    cost_impact.append(delta)
    print("OSE relative validation cost (",i+1,"/",x.shape[1],"):", new_cost)

cost_impact = np.array(cost_impact).reshape(-1, 1)/perturbation





'''Plot Costs'''
warnings.filterwarnings("ignore")
plt.plot(nnet.costs)
plt.show()


plt.plot(cost_impact, label="OSE")
plt.plot(sensitivity, label="FSO")
plt.legend()
plt.show()
