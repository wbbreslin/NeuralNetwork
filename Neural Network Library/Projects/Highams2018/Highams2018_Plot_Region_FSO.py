import numpy as np
import Highams2018 as hig
import matplotlib.pyplot as plt
from Data import data
from Highams2018_FSO import sensitivity
from matplotlib.colors import LinearSegmentedColormap

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

xx = hig.x
yy = hig.y

# Separate the data into two sets based on Y values
x_y0 = df.x[(df.y[:, 0] == 1) & (df.y[:, 1] == 0)]
x_y1 = df.x[(df.y[:, 0] == 0) & (df.y[:, 1] == 1)]

# Create a scatterplot for Y=0 points
predicted_failure = plt.scatter(x_y0[:, 0], x_y0[:, 1], c='bisque', label='Predicted Failure', marker='s')

# Create a scatterplot for Y=1 points
predicted_success = plt.scatter(x_y1[:, 0], x_y1[:, 1], c='palegreen', label='Predicted Success', marker='s')

# Plot the validation data
xv = hig.x_validation
yv = hig.y_validation

#vF = plt.scatter(xv[0:2, 0], xv[0:2, 1],  marker='x', s=10, color = 'red', label = 'Validation Data - Failure')
#vS = plt.scatter(xv[2:4, 0], xv[2:4, 1], marker='o', s=10, color = 'red', label = 'Validation Data - Success')

# Set color of points based on sensitivity values
#color_vector = abs(sensitivity)

def grayscale_diverging_cmap():
    cmap = LinearSegmentedColormap.from_list(
        'grayscale_diverging', [(0, 'black'), (0.5, 'white'), (1, 'black')], N=256
    )
    return cmap

color_vector = sensitivity
print(sensitivity)
#cmap = grayscale_diverging_cmap()
cmap = plt.get_cmap('seismic')
vmax = np.max(np.abs(color_vector))

# Scatterplot for original data
scatter_failure = plt.scatter(xx[0:5, 0], xx[0:5, 1], label='Failure', marker='s', s=75, c = color_vector[0:5], cmap = cmap, edgecolor = "black",vmin=-vmax,vmax= vmax)
scatter_success = plt.scatter(xx[5:10, 0], xx[5:10, 1], c = color_vector[5:10], label='Success', marker='o', s=75, cmap = cmap, edgecolor = "black",vmin=-vmax,vmax= vmax)


cbar = plt.colorbar()

cbar.set_label('Sensitivity')
plt.title('Forecast Sensitivity to Data Removal')
# Create proxy artists for the legend with black edge and fill colors
legend_failure = plt.Line2D([0], [0], marker='s', color='black', markerfacecolor='black', markersize=10, label='Failure', linestyle='None')
legend_success = plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='black', markersize=10, label='Success', linestyle='None')


# Add the proxy artists to the legend without affecting the actual points
plt.legend(handles=[predicted_failure, predicted_success, legend_failure, legend_success], loc=2)
plt.show()