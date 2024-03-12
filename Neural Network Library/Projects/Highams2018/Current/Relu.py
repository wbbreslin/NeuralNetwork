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
x = np.array([[0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7],
              [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]]).T

y = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]).T

x_validation = np.array([[0.7, 0.2, 0.6, 0.9],
                         [0.9, 0.7, 0.1, 0.8]]).T

y_validation = np.array([[1, 1, 0, 0],
                         [0, 0, 1, 1]]).T

'''
x_validation = np.array([[0.5,0.1,0.2,0.7,0.2,0.6,0.9,0.8,0.6,0.8],
                         [0.1,0.9,0.3,0.9,0.7,0.1,0.8,0.4,0.6,0.1]]).T

y_validation = np.array([[1,1,1,1,1,0,0,0,0,0],
                         [0,0,0,0,0,1,1,1,1,1]]).T
'''

'''Define the model'''
np.random.seed(333)
training = data(x, y)
n = x.shape[0]
validation = data(x_validation, y_validation)
#theta = 0.00001 / n
theta = 0
nnet = neural_network(layers=[2, 2, 2],
                      activation_functions=[af.sigmoid,
                                            af.sigmoid],
                      cost_function=cf.half_SSE,
                      regularization=theta)

'''Duplicate untrained model for OSE'''
nnet_OSE_init = copy.deepcopy(nnet)

'''Train to optimality'''
itr1 = 10000;
step1 = 0.25  # 6000 and 0.25
itr2 = 10000;
step2 = 0.05

nnet.train(training, max_iterations=itr1, step_size=step1)
nnet.train(training, max_iterations=itr2, step_size=step2)

plt.plot(nnet.costs)
plt.show()

nnet.predict(training)
print(np.round(training.predictions))


def ordered_pair_matrix(start, end, step):
    x = np.arange(start, end + step, step)
    y = np.arange(start, end + step, step)
    xx, yy = np.meshgrid(x, y)
    matrix = np.column_stack((xx.ravel(), yy.ravel()))
    return matrix

xx = x
yy = y

x = ordered_pair_matrix(0,1,0.01)
df = data(x,y=None)
nnet.predict(df)
df.y = np.round(df.predictions)



# Separate the data into two sets based on Y values
x_y0 = df.x[(df.y[:, 0] == 1) & (df.y[:, 1] == 0)]
x_y1 = df.x[(df.y[:, 0] == 0) & (df.y[:, 1] == 1)]

# Create a scatterplot for Y=0 points
predicted_failure = plt.scatter(x_y0[:, 0], x_y0[:, 1], c='bisque', label='Predicted Failure', marker='s')

# Create a scatterplot for Y=1 points
predicted_success = plt.scatter(x_y1[:, 0], x_y1[:, 1], c='palegreen', label='Predicted Success', marker='s')

# Plot the validation data
xv = validation.x
yv = validation.y

# Scatterplot for original data
scatter_failure = plt.scatter(xx[0:5, 0], xx[0:5, 1], label='Training Data - Failure', marker='x', s=75, edgecolor = "black", c="black")
scatter_success = plt.scatter(xx[5:10, 0], xx[5:10, 1], label='Training Data - Success', marker='o', s=75, edgecolor = "black", c="black")

plt.title('Training Data')

# Add the proxy artists to the legend without affecting the actual points
plt.legend(loc=2)
plt.show()