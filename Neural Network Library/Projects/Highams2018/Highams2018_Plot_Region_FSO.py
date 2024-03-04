import numpy as np
import Highams2018 as hig
import matplotlib.pyplot as plt
from Data import data
from Highams2018_FSO import sensitivity

def ordered_pair_matrix(start, end, step):
    x = np.arange(start, end + step, step)
    y = np.arange(start, end + step, step)
    xx, yy = np.meshgrid(x, y)
    matrix = np.column_stack((xx.ravel(), yy.ravel()))
    return matrix

nnet = hig.nnet_copy
x = ordered_pair_matrix(0,1,0.01)
df = data(x,y=None)
nnet.predict(df)
df.y = np.round(nnet.predictions)
print(df.x.shape)
print(df.y.shape)

xx = hig.x
yy = hig.y

# Separate the data into two sets based on Y values
x_y0 = df.x[(df.y[:, 0] == 1) & (df.y[:, 1] == 0)]
x_y1 = df.x[(df.y[:, 0] == 0) & (df.y[:, 1] == 1)]

# Create a scatterplot for Y=0 points
plt.scatter(x_y0[:, 0], x_y0[:, 1], c='bisque', label='Predicted Failure', marker='s')

# Create a scatterplot for Y=1 points
plt.scatter(x_y1[:, 0], x_y1[:, 1], c='palegreen', label='Predicted Success', marker='s')

# Set color of points based on sensitivity values
color_vector = abs(sensitivity)
cmap = plt.get_cmap('PuOr')

# Scatterplot for original data
plt.scatter(xx[0:5, 0], xx[0:5, 1], label='Failure', marker='x', s=75, c = color_vector[0:5], cmap = cmap)
plt.scatter(xx[5:10, 0], xx[5:10, 1], c = color_vector[5:10], label='Success', marker='o', s=75, cmap = cmap)

cbar = plt.colorbar()
cbar.set_label('Sensitivity')
plt.title('Forecast Sensitivity to Data Removal')
plt.show()