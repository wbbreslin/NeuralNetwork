import Highams2019 as data
import numpy as np
import torch
import TrainingAlgorithms as train

nnet = data.nnet

'''Define the initial weights'''
w0 = np.array([[0.1, 0.2],
               [0.3, 0.4],
               [0.5, 0.6]])

w1 = np.array([[0.7, 0.8, 0.9],
               [1.0, 1.1, 1.2],
               [1.3, 1.4, 1.5]])

w2 = np.array([[1.6, 1.7],
               [1.8, 1.9],
               [2.0, 2.1],
               [2.2, 2.3]])

'''Use these custom weights to initialize the network'''
weights = [w0, w1, w2]
nnet['Weights']=weights

print(nnet.keys())