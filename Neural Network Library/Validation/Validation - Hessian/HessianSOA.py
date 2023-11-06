import numpy as np
import Base as base
import FirstOrderModel as fom
import TrainingAlgorithms as train
import SecondOrderModel as som


"""The data set of predictor variables"""
x_predictors = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],
                        [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]]).T

"""The data set of outcome variables"""
y_outcomes = np.array([[1,1,1,1,1,0,0,0,0,0],
                      [0,0,0,0,0,1,1,1,1,1]]).T

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

weights = [w0, w1, w2]

'''Create a neural network object'''
nnet = {'Predictors': x_predictors,
        'Outcomes': y_outcomes,
        'Weights': weights}

'''Exact gradient calculation'''
training_itr = 4000
nnet = train.gradient_descent(nnet,max_iterations=training_itr)

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
print(nnet['Hv_Products'][0])