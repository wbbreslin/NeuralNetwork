import numpy as np
from Data import data
import matplotlib.pyplot as plt

def region_plot(nnet, df):
    x_region = ordered_pair_matrix(0,1,0.01)
    df_region = data(x_region,y=None)
    nnet.predict(df_region)
    df_region.y = np.round(nnet.predictions)

    # Separate the data into two sets based on Y values
    x_y0 = df_region.x[(df_region.y[:, 0] == 1) & (df_region.y[:, 1] == 0)]
    x_y1 = df_region.x[(df_region.y[:, 0] == 0) & (df_region.y[:, 1] == 1)]

    fig = plt.figure()

    # Create a scatterplot for predicted region
    plt.scatter(x_y0[:, 0], x_y0[:, 1], c='bisque', label='Predicted Failure', marker='s')
    plt.scatter(x_y1[:, 0], x_y1[:, 1], c='palegreen', label='Predicted Success', marker='s')

    # Scatterplot for original data
    index = df.y[:, 0]
    plt.scatter(df.x[index==1, 0], df.x[index==1, 1], label='Failure', marker='x', s=75, c="black")
    plt.scatter(df.x[index==0, 0], df.x[index==0, 1], label='Success', marker='o', s=75,c="black")

    # Add the proxy artists to the legend without affecting the actual points
    plt.legend(loc=2)
    return fig

def ordered_pair_matrix(start, end, step):
    x = np.arange(start, end + step, step)
    y = np.arange(start, end + step, step)
    xx, yy = np.meshgrid(x, y)
    matrix = np.column_stack((xx.ravel(), yy.ravel()))
    return matrix