import numpy as np
import pandas as pd
from Data import data
import NeuralNetwork as nn
import ActivationFunctions as af
import CostFunctions as cf
import matplotlib.pyplot as plt

np.random.seed(314)

'''Import and preview the data'''
file_path = 'iris.csv'
df = pd.read_csv(file_path)
print(df.head())

'''Data cleaning'''
x_columns = df.iloc[:,:4]
y_columns = df['species']
x = x_columns.to_numpy()
y = y_columns.to_numpy()
unique_species, indices = np.unique(y, return_inverse=True)
unit_vectors = np.eye(len(unique_species))
y = unit_vectors[indices]
print("X dimensions:", x.shape)
print("Y dimensions:", y.shape)

iris = data(x,y)
training, validation = iris.test_train_split(train_percent=0.8)
training.x = training.x.T
training.y = training.y.T
validation.x = validation.x.T
validation.y = validation.y.T
print(training.x.shape)
print(validation.x.shape)

nnet = nn.neural_network(layers=[4,8,8,3],
                           activation_functions = [af.sigmoid,
                                                   af.sigmoid,
                                                   af.softmax],
                           cost_function=cf.half_SSE)

nnet.train(training, max_iterations =1000, step_size=0.05)
nnet.train(training, max_iterations =200, step_size=0.01)
plt.plot(nnet.costs)
plt.show()
print(nnet.costs[-1])