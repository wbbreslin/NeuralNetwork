import numpy as np
import Base as base
import SecondOrderModel as som
import TrainingAlgorithms as train
import SIAM2019 as data

'''Import the SIAM2019 data and network structure'''
nnet = data.nnet.copy()

'''Train the model'''
nnet = train.gradient_descent(nnet,max_iterations=10**4)

'''Define unit vectors to get full hessian'''
'''Focusing on W0 to start'''
n = nnet['Predictors'].shape[0]
p = nnet['Augmented_Weights'][0].shape[0]
q = nnet['Augmented_Weights'][0].shape[1]
identity = np.eye((p+1)*q)
vector = identity[:,0].reshape((p+1)*q,1)
vectors = [vector, np.zeros((9,1)), np.zeros((8,1))]

nnet = som.forward_pass(nnet, vectors)
nnet = som.backward_pass(nnet, vectors)
