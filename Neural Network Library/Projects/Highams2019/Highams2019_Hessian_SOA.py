import Highams2019_Train_GradientDescent as data
import numpy as np
import Base as base
import SecondOrderModel as som

'''Import the neural network and the data set'''
nnet = data.nnet

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

print(full_hessian[:,0])
#delta = full_hessian - full_hessian.T
#cn = np.linalg.cond(full_hessian)
#print(cn)