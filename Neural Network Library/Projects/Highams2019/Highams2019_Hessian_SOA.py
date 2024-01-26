import Highams2019_Train_GradientDescent as data
import numpy as np
import Base as base
import SecondOrderModel as som

'''Import the neural network and the data set'''
nnet = data.nnet

'''Full Hessian'''
def hessian(nnet):
    weights = nnet['Weights']
    elements = [w.size for w in weights]
    partitions = np.append(0,np.cumsum(elements))
    dimensions = partitions[-1]
    full_hessian = np.zeros((dimensions,dimensions))

    for i in range(dimensions):
        vector = np.zeros((dimensions,1))
        vectors = []
        vector[i]=1
        for j in range(len(weights)):
            v = vector[partitions[j]:partitions[j+1]]
            vectors.append(v)
        nnet = som.forward_pass(nnet, vectors)
        nnet = som.backward_pass(nnet, vectors)
        columns = []
        for k in range(len(weights)):
            hvp = base.to_vector(nnet['Hv_Products'][k])
            columns.append(hvp)
        column_hessian = columns[0]
        for c in range(len(columns)-1):
            column_hessian = np.vstack((column_hessian,columns[c+1]))
        full_hessian[:,i] = column_hessian[:,0]
    return(full_hessian)

full_hessian = hessian(nnet)
det = np.linalg.det(full_hessian)
print(det)
