import numpy as np
import Base as base
import TrainingAlgorithms as train
import FirstOrderModel as fom
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

# Define your data
x_predictors = np.array([[0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7],
                        [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]]).T

y_outcomes = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]).T

"""Define the neural network structure"""
np.random.seed(100)
neurons = np.array([2,2,3,2])
activations = ["sigmoid","sigmoid","sigmoid","sigmoid"]
weights, biases = base.create_network(neurons)
weights = base.augment_network(weights, biases)

nnet = {'Predictors': x_predictors,
        'Outcomes': y_outcomes,
        'Weights': weights,
        'Neurons': neurons}

nnet = train.gradient_descent(nnet,max_iterations=4000)
def ordered_pair_matrix(start, end, step):
    x = np.arange(start, end + step, step)
    y = np.arange(start, end + step, step)
    xx, yy = np.meshgrid(x, y)
    matrix = np.column_stack((xx.ravel(), yy.ravel()))
    return matrix


x = ordered_pair_matrix(0,1,0.01) # do not make any bigger...
nnet['Predictors'] = x
y = fom.forward_pass(nnet)
y = np.round(nnet['States'][-1])


# Separate the data into two sets based on Y values
x_y0 = x[(y[:, 0] == 1) & (y[:, 1] == 0)]
x_y1 = x[(y[:, 0] == 0) & (y[:, 1] == 1)]

# Create a scatterplot for Y=0 points
plt.scatter(x_y0[:, 0], x_y0[:, 1], c='bisque', label='Predicted Failure', marker='s')

# Create a scatterplot for Y=1 points
plt.scatter(x_y1[:, 0], x_y1[:, 1], c='palegreen', label='Predicted Success', marker='s')

# Compute and plot the convex hull around Y=1 points with a filled interior
hull = ConvexHull(x_y1)
polygon = Polygon(x_y1[hull.vertices], closed=True, facecolor='palegreen', alpha=0.3)
plt.gca().add_patch(polygon)

x1 = x_predictors[y_outcomes[:, 0] == 1]
x2 = x_predictors[y_outcomes[:, 1] == 1]

plt.scatter(x1[:, 0], x1[:, 1], label='Failure', marker='x', s=75, color = 'black')
plt.scatter(x2[:, 0], x2[:, 1], color = 'black', label='Success', marker='o', s=75)

#plt.xlim(0, 1)  # Replace x_min and x_max with your desired values
#plt.ylim(0, 1)

plt.xlabel('X1 - Axis')
plt.ylabel('X2 - Axis')
plt.title('Neural Network Predictions (Iterations: None)')
plt.legend(loc=2)
plt.grid(True)
plt.show()


print(np.sum(y))
print(y.shape)