import numpy as np
from Highams2018 import nnet, df
from NNET import neural_network, data
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

def ordered_pair_matrix(start, end, step):
    x = np.arange(start, end + step, step)
    y = np.arange(start, end + step, step)
    xx, yy = np.meshgrid(x, y)
    matrix = np.column_stack((xx.ravel(), yy.ravel()))
    return matrix

nnet = highams.nnet
df = ordered_pair_matrix(0,1,0.01) # do not make any bigger...
nnet['Pass Forward'] = x
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

x1 = nnet['Predictors'][nnet['Outcomes'][:, 0] == 1]
x2 = nnet['Predictors'][nnet['Outcomes'][:, 1] == 1]

plt.scatter(x1[:, 0], x1[:, 1], label='Failure', marker='x', s=75, color = 'black')
plt.scatter(x2[:, 0], x2[:, 1], color = 'black', label='Success', marker='o', s=75)

plt.xlabel('X1 - Axis')
plt.ylabel('X2 - Axis')
plt.title('Example Data Set')
plt.xlim(0,1)
plt.ylim(0,1)
plt.legend(loc=2)
plt.grid(True)
plt.show()