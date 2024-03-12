import numpy as np
from NNET import neural_network
from Data import data
import ActivationFunctions as af
import CostFunctions as cf
import copy
import matplotlib.pyplot as plt
import Base as base
from scipy.stats import spearmanr
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap

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
validation = data(x_validation,y_validation)
theta = 0.001/n
nnet = neural_network(layers=[2,3,2],
                      activation_functions = [af.linear,
                                              af.sigmoid],
                      cost_function=cf.half_SSE,
                      regularization = theta)


'''Duplicate untrained model for OSE'''
nnet_OSE_init = copy.deepcopy(nnet)

'''Train to optimality'''
itr1 = 5000 ; step1 = 0.25 #6000 and 0.25
itr2 = 5000; step2 = 0.05

nnet.train(training, max_iterations=itr1, step_size=step1)
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

#print(nnet_FSO_q.gradient_vector)

'''OSE Sensitivity Analysis'''
#Training - Compute Unmodified Validation Cost
nnet_OSE = copy.deepcopy(nnet_OSE_init)
nnet_OSE.train(training, max_iterations = itr1, step_size=step1)
nnet_OSE.train(training, max_iterations = itr2, step_size=step2)
nnet_OSE_validation = copy.deepcopy(nnet_OSE)
nnet_OSE_validation.forward(validation)
#nnet_OSE_validation.backward(validation)
nnet_OSE_validation.track_cost(validation)
unmodified_cost = nnet_OSE_validation.costs[-1]

ref_weights = [base.to_vector(w) for w in nnet_OSE.weights]
ref_weights = np.vstack(ref_weights)


cost_impact = []
perturbation = 10**-6
#with step1=0.25 and step2=0.05
#perturbation = (step1*itr1 + step2*itr2)/(itr1+itr2)
for i in range(x.shape[0]):
    nnet_OSE = copy.deepcopy(nnet_OSE_init)
    training_OSE = copy.deepcopy(training)
    training_OSE.s = np.ones((x.shape[0],1))
    training_OSE.s[i] = 1-perturbation
    nnet_OSE.train(training_OSE, max_iterations=itr1, step_size=step1)
    nnet_OSE.train(training_OSE, max_iterations=itr2, step_size=step2)
    nnet_OSE_validation = copy.deepcopy(nnet_OSE)
    nnet_OSE_validation.forward(validation)
    #nnet_OSE_validation.backward(validation)
    nnet_OSE_validation.track_cost(validation)
    new_cost = nnet_OSE_validation.costs[-1]
    delta = new_cost-unmodified_cost
    cost_impact.append(delta)

cost_impact = np.array(cost_impact).reshape(-1, 1)/perturbation

'''Plot OSE Results'''
def ordered_pair_matrix(start, end, step):
    x = np.arange(start, end + step, step)
    y = np.arange(start, end + step, step)
    xx, yy = np.meshgrid(x, y)
    matrix = np.column_stack((xx.ravel(), yy.ravel()))
    return matrix

def grayscale_diverging_cmap():
    cmap = LinearSegmentedColormap.from_list(
        'grayscale_diverging', [(0, 'black'), (0.5, 'white'), (1, 'black')], N=256
    )
    return cmap

x_region = ordered_pair_matrix(0,1,0.01)
df_region = data(x_region,y=None)
nnet.predict(df_region)
df_region.y = np.round(df_region.predictions)

# Separate the data into two sets based on Y values
x_y0 = df_region.x[(df_region.y[:, 0] == 1) & (df_region.y[:, 1] == 0)]
x_y1 = df_region.x[(df_region.y[:, 0] == 0) & (df_region.y[:, 1] == 1)]

width_ratios = [1, 1, 1]
height_ratios = [1]

figure, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'width_ratios': width_ratios, 'height_ratios': height_ratios})

# Plot for Predicted Region with colorbar
scatter1 = axes[0].scatter(x_y0[:, 0], x_y0[:, 1], c='bisque', label='Predicted Failure', marker='s')
scatter2 = axes[0].scatter(x_y1[:, 0], x_y1[:, 1], c='palegreen', label='Predicted Success', marker='s')

index = training.y[:, 0]
color_vector = np.array(cost_impact).reshape(-1, 1)
cmap = plt.get_cmap('seismic')
#cmap = grayscale_diverging_cmap()
#vmax = np.max(color_vector)
#vmin = np.min(color_vector)
vmax = np.max(abs(color_vector))
vmin = -vmax

scatter3 = axes[0].scatter(training.x[index == 1, 0], training.x[index == 1, 1], label='Failure', marker='s', s=75, c=color_vector[index == 1], cmap=cmap, vmin=vmin, vmax=vmax, edgecolor="black")
scatter4 = axes[0].scatter(training.x[index == 0, 0], training.x[index == 0, 1], label='Success', marker='o', s=75, c=color_vector[index == 0], cmap=cmap, vmin=vmin, vmax=vmax, edgecolor="black")
scatter_v11 = axes[0].scatter(validation.x[0:2,0],validation.x[0:2,1], marker = "s", c="black", s=25, edgecolor="black")
scatter_v12 = axes[0].scatter(validation.x[2:4,0],validation.x[2:4,1], marker = "o", c="black", s=25, edgecolor="black")
axes[0].legend(loc=2)
axes[0].set_aspect('equal', adjustable='box')
axes[0].set_title("OSE")

for i, txt in enumerate(range(len(training.x))):
    axes[0].annotate(txt, (training.x[i, 0], training.x[i, 1]), textcoords="offset points", xytext=(0, 8), ha='center')

# Create colorbar for the first plot
sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])  # dummy array for the scalar mappable
cbar = figure.colorbar(sm, ax=axes[0], shrink=0.6)
#cbar.set_label('Sensitivity')

# Plot for Predicted Region with colorbar
scatter5 = axes[1].scatter(x_y0[:, 0], x_y0[:, 1], c='bisque', label='Predicted Failure', marker='s')
scatter6 = axes[1].scatter(x_y1[:, 0], x_y1[:, 1], c='palegreen', label='Predicted Success', marker='s')

index = training.y[:, 0]
color_vector = np.array(sensitivity)
cmap = plt.get_cmap('seismic')
#vmax = np.max(sensitivity)
#vmin = np.min(sensitivity)
#cmap = plt.get_cmap('Greys')
#cmap = grayscale_diverging_cmap()
vmax = np.max(abs(color_vector))
vmin = -vmax

scatter7 = axes[1].scatter(training.x[index == 1, 0], training.x[index == 1, 1], label='Failure', marker='s', s=75, c=color_vector[index == 1], cmap=cmap, vmin=vmin, vmax=vmax, edgecolor="black")
scatter8 = axes[1].scatter(training.x[index == 0, 0], training.x[index == 0, 1], label='Success', marker='o', s=75, c=color_vector[index == 0], cmap=cmap, vmin=vmin, vmax=vmax, edgecolor="black")
scatter_v21 = axes[1].scatter(validation.x[0:2,0],validation.x[0:2,1], marker = "s", c="black", s=25, edgecolor="black")
scatter_v22 = axes[1].scatter(validation.x[2:4,0],validation.x[2:4,1], marker = "o", c="black", s=25, edgecolor="black")
axes[1].legend(loc=2)
axes[1].set_title("FSO")
axes[1].set_aspect('equal', adjustable='box')

for i, txt in enumerate(range(len(training.x))):
    axes[1].annotate(txt, (training.x[i, 0], training.x[i, 1]), textcoords="offset points", xytext=(0, 8), ha='center')

# Create colorbar for the second plot
sm2 = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm2.set_array([])  # dummy array for the scalar mappable
cbar2 = figure.colorbar(sm2, ax=axes[1], shrink=0.6)
#cbar2.set_label('Sensitivity')

#coefficients = np.polyfit(cost_impact.flatten(), sensitivity.flatten(),1)
#print(coefficients)

axes[2].plot(cost_impact, label="OSE")
axes[2].plot(sensitivity, label="FSO")
axes[2].legend()

axes[2].set_aspect('auto')
axes[2].set_box_aspect(0.8)
axes[2].set_title("Sensitivity Comparison")
plt.tight_layout()
plt.show()

print(np.corrcoef(cost_impact.flatten(),sensitivity.flatten()))
correlation_coefficient, p_value = spearmanr(cost_impact.flatten(), sensitivity.flatten())
print(correlation_coefficient)

print(np.hstack((cost_impact,sensitivity, cost_impact-sensitivity)))