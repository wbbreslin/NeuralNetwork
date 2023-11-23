import Highams2019 as data
import numpy as np
import Base as base
import TrainingAlgorithms as train
import SecondOrderModel as som

'''Import the neural network and the data set'''
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

'''Exact gradient calculation'''
training_itr = 4000

nnet = train.gradient_descent(nnet,max_iterations=training_itr)


'''Full Hessian'''
full_hessian = np.zeros((23,23))

for i in range(23):
    vector = np.zeros((23,1))
    vector[i]=1
    v1 = vector[0:6]
    v2 = vector[6:15]
    v3 = vector[15:23]
    vectors = [v1, v2, v3]
    nnet = som.forward_pass(nnet, vectors)
    nnet = som.backward_pass(nnet, vectors)
    H0 = base.to_vector(nnet['Hv_Products'][0])
    H1 = base.to_vector(nnet['Hv_Products'][1])
    H2 = base.to_vector(nnet['Hv_Products'][2])
    column = np.vstack((H0, H1, H2))
    full_hessian[:,i] = column[:,0]

#print(full_hessian)
#delta = full_hessian - full_hessian.T
#cn = np.linalg.cond(full_hessian)
#print(cn)