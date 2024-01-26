import FirstOrderModel as fom
import SecondOrderModel as som
import TrainingAlgorithms as train
import numpy as np
import Base as base

"""The data set of predictor variables"""
x_predictors = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],
                        [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]]).T

"""The data set of outcome variables"""
y_outcomes = np.array([[1,1,1,1,1,0,0,0,0,0],
                      [0,0,0,0,0,1,1,1,1,1]]).T

"""Define the neural network structure"""
np.random.seed(100)
nnet = base.create_network(x_predictors,
                           y_outcomes,
                           neurons = [2,2,3,2],
                           activations = [base.sigmoid,
                                          base.sigmoid,
                                          base.sigmoid])

def prune_network(nnet):
    #Initialization
    weights = nnet['Weights']
    elements = [w.size for w in weights]
    partitions = np.append(0, np.cumsum(elements))
    pruning_matrices = [np.ones(x.shape) for x in weights]

    #Training
    nnet = train.gradient_descent(nnet, step_size=0.25, max_iterations=4000, pruning_matrices=pruning_matrices)

    #Saliency Analysis
    s = saliency(nnet)
    index_of_smallest_non_zero = np.argmin(data[data != 0])


    return(nnet)

def saliency(nnet):
    weights = nnet['Weights']
    vec_weights = [base.to_vector(w) for w in weights]
    vec_weights = np.vstack(vec_weights)
    hessian = hessian_matrix(nnet)
    inverse_hessian = np.linalg.inv(hessian)
    diagonals = np.array(np.diag(inverse_hessian))
    diagonals = diagonals.reshape((len(vec_weights),1))
    saliency = vec_weights**2 / (2 * abs(diagonals))
    return(saliency)

def hessian_matrix(nnet):
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


nnet = prune_network(nnet)
s = saliency(nnet)
print(s)
print(np.percentile(s,20))
