import numpy as np
import Base as base
import FirstOrderModel as fom
import SecondOrderModel as som
import TrainingAlgorithms as train
import copy
import matplotlib.pyplot as plt

"""Define Finite Difference Hessian as a Function of Epsilon"""
def FDM_Hessian(nnet, epsilon):
    nnet_plus = copy.deepcopy(nnet)
    nnet_minus = copy.deepcopy(nnet)
    Hessians = []
    for u in range(len(nnet['Weights'])):
        for w in range(len(nnet['Weights'])):
            shape_num = nnet['Weights'][u].shape
            shape_den = nnet['Weights'][w].shape
            p1 = shape_num[0]
            q1 = shape_num[1]
            p2 = shape_den[0]
            q2 = shape_den[1]
            Hessian = np.zeros((p1 * q1, p2 * q2))
            iterate = 0
            for j in range(q2):
                for i in range(p2):
                    perturbation = np.zeros((p2, q2))
                    perturbation[i, j] = epsilon
                    nnet_minus['Weights'][w] = nnet_minus['Weights'][w] - perturbation
                    nnet_plus['Weights'][w] = nnet_plus['Weights'][w] + perturbation

                    nnet_minus = fom.forward_pass(nnet_minus)
                    nnet_minus = fom.backward_pass(nnet_minus)

                    nnet_plus = fom.forward_pass(nnet_plus)
                    nnet_plus = fom.backward_pass(nnet_plus)

                    delta = (nnet_plus['Gradients'][u] - nnet_minus['Gradients'][u]) / (2 * epsilon)
                    Hessian_column, dim = base.to_vector(delta)
                    Hessian[:, iterate] = Hessian_column[:, 0]
                    iterate = iterate + 1

                    '''Reset weights'''
                    nnet_minus['Weights'][w] = nnet_minus['Weights'][w] + perturbation
                    nnet_plus['Weights'][w] = nnet_plus['Weights'][w] - perturbation

            Hessians.append(Hessian)

    H = Hessians
    full_hessian = np.bmat([[H[0], H[1], H[2]],
                            [H[3], H[4], H[5]],
                            [H[6], H[7], H[8]]])
    return full_hessian


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

'''Exact Hessian calculation'''
'''Full Hessian'''
exact_hessian = np.zeros((23,23))

for i in range(23):
    vector = np.zeros((23,1))
    vector[i]=1
    v1 = vector[0:6]
    v2 = vector[6:15]
    v3 = vector[15:23]
    vectors = [v1, v2, v3]
    nnet = som.forward_pass(nnet, vectors)
    nnet = som.backward_pass(nnet, vectors)
    H0, d0 = base.to_vector(nnet['Hv_Products'][0])
    H1, d1 = base.to_vector(nnet['Hv_Products'][1])
    H2, d2 = base.to_vector(nnet['Hv_Products'][2])
    column = np.vstack((H0, H1, H2))
    exact_hessian[:,i] = column[:,0]

approx_hessian = FDM_Hessian(nnet, 10**-6)

w = np.ones((23,1))
#numerator = w.T @ approx_hessian @ w
#denominator = w.T @ exact_hessian @ w
#print(numerator/denominator)

"""Plot the Difference"""
delta = exact_hessian - approx_hessian
min = np.abs(np.min(delta))
max = np.abs(np.max(delta))
bound = np.max((min, max))
plt.imshow(delta, cmap='seismic', vmin=-bound, vmax=bound)
plt.colorbar()
plt.xlabel('Weight Parameter ID (23 parameters)')
plt.ylabel('Weight Parameter ID (23 parameters)')
plt.title('Hessian Matrix: ' + str(training_itr) + ' Iterations')
plt.show()