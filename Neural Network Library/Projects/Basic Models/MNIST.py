import numpy as np
import pandas as pd
from Data import data


'''Import and preview the data'''
training_file= '../Datasets/mnist_train.csv'
validation_file= '../Datasets/mnist_test.csv'
df_training = pd.read_csv(training_file)
df_validation = pd.read_csv(training_file)
print(df_training.head())

'''Data cleaning'''
x_training = df_training.iloc[:,1:]
y_training = df_training.iloc[:,0]
x_training = x_training.to_numpy()
unique_labels, indices = np.unique(y_training, return_inverse=True)
unit_vectors = np.eye(len(unique_labels))
y_training = unit_vectors[indices]
training = data(x_training,y_training)

x_validation = df_validation.iloc[:,1:]
y_validation = df_validation.iloc[:,0]
x_validation = x_validation.to_numpy()
unique_labels, indices = np.unique(y_validation, return_inverse=True)
unit_vectors = np.eye(len(unique_labels))
y_validation = unit_vectors[indices]
validation = data(x_validation,y_validation)

print("X dimensions:", training.x.shape)
print("Y dimensions:", training.y.shape)

