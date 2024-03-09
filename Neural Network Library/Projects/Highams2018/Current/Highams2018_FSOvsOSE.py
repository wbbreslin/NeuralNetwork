import numpy as np
from NNET import neural_network
from Data import data
import ActivationFunctions as af
import CostFunctions as cf
import copy
import matplotlib.pyplot as plt
import Base as base
from scipy.stats import spearmanr

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
np.random.seed(333)
training = data(x,y)
n = x.shape[0]
theta = 0.001/n
validation = data(x_validation,y_validation)
nnet = neural_network(layers=[2,3,2],
                      activation_functions = [af.sigmoid,
                                              af.sigmoid],
                      cost_function=cf.half_SSE,
                      regularization = theta)


'''Duplicate untrained model for OSE'''
nnet_OSE_init = copy.deepcopy(nnet)

'''Train to optimality'''
itr1 = 6000 ; step1 = 0.25
itr2 = 4000; step2 = 0.05

nnet.train(training, max_iterations=itr1, step_size=0.25)
nnet.train(training, max_iterations=itr2, step_size=0.05)

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

#print(nnet_FSO_q.gradient_vector)

'''OSE Sensitivity Analysis'''
#Training - Compute Unmodified Validation Cost
nnet_OSE = copy.deepcopy(nnet_OSE_init)
nnet_OSE.train(training, max_iterations = itr1, step_size=0.25)
nnet_OSE.train(training, max_iterations = itr2, step_size=0.05)
nnet_OSE_validation = copy.deepcopy(nnet_OSE)
nnet_OSE_validation.forward(validation)
#nnet_OSE_validation.backward(validation)
nnet_OSE_validation.track_cost(validation)
unmodified_cost = nnet_OSE_validation.costs[-1]

ref_weights = [base.to_vector(w) for w in nnet_OSE.weights]
ref_weights = np.vstack(ref_weights)


cost_impact = []
for i in range(x.shape[0]):
    nnet_OSE = copy.deepcopy(nnet_OSE_init)
    training_OSE = copy.deepcopy(training)
    training_OSE.s = np.ones((x.shape[0],1))
    training_OSE.s[i] = 0.8
    nnet_OSE.train(training_OSE, max_iterations=itr1, step_size=0.25)
    nnet_OSE.train(training_OSE, max_iterations=itr2, step_size=0.05)
    nnet_OSE_validation = copy.deepcopy(nnet_OSE)
    nnet_OSE_validation.forward(validation)
    #nnet_OSE_validation.backward(validation)
    nnet_OSE_validation.track_cost(validation)
    new_cost = nnet_OSE_validation.costs[-1]
    delta = new_cost-unmodified_cost
    cost_impact.append(delta)

cost_impact = np.array(cost_impact).reshape(-1,1)
plt.plot(cost_impact, label="OSE")
plt.plot(sensitivity, label="FSO")
plt.legend()
plt.show()

print(np.corrcoef(cost_impact.flatten(),sensitivity.flatten()))
correlation_coefficient, p_value = spearmanr(cost_impact.flatten(), sensitivity.flatten())
print(correlation_coefficient)


'''Plot OSE Results'''
def ordered_pair_matrix(start, end, step):
    x = np.arange(start, end + step, step)
    y = np.arange(start, end + step, step)
    xx, yy = np.meshgrid(x, y)
    matrix = np.column_stack((xx.ravel(), yy.ravel()))
    return matrix

x_region = ordered_pair_matrix(0,1,0.01)
df_region = data(x_region,y=None)
nnet.predict(df_region)
df_region.y = np.round(df_region.predictions)

# Separate the data into two sets based on Y values
x_y0 = df_region.x[(df_region.y[:, 0] == 1) & (df_region.y[:, 1] == 0)]
x_y1 = df_region.x[(df_region.y[:, 0] == 0) & (df_region.y[:, 1] == 1)]

#fig = plt.figure()

# Create a scatterplot for predicted region
plt.scatter(x_y0[:, 0], x_y0[:, 1], c='bisque', label='Predicted Failure', marker='s')
plt.scatter(x_y1[:, 0], x_y1[:, 1], c='palegreen', label='Predicted Success', marker='s')

# Scatterplot for original data
index = training.y[:, 0]
color_vector = np.array(cost_impact).reshape(-1,1)
cmap = plt.get_cmap('Greys')
vmax = np.max(color_vector)
vmin = np.min(color_vector)
plt.scatter(training.x[index==1, 0], training.x[index==1, 1], label='Failure', marker='s', s=75, c=color_vector[index==1], cmap=cmap, vmin=vmin, vmax= vmax, edgecolor="black")
plt.scatter(training.x[index==0, 0], training.x[index==0, 1], label='Success', marker='o', s=75,c=color_vector[index==0], cmap=cmap, vmin=vmin, vmax= vmax, edgecolor="black")

cbar = plt.colorbar()
cbar.set_label('Sensitivity')

# Add the proxy artists to the legend without affecting the actual points
plt.legend(loc=2)
plt.show()
