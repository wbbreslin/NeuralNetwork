import numpy as np
import pandas as pd


'''Import and preview the data'''
file_path = '../Datasets/iris.csv'
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