import SecondOrderModel as som
import TrainingAlgorithms as train
import numpy as np
import Base as base


def single_step_prune(nnet, remove):
    #Initialization
    weights = nnet['Weights']
    pruning_matrices = [np.ones(x.shape) for x in weights]

    #Training
    nnet = train.gradient_descent(nnet, step_size=0.25, max_iterations=4000, pruning_matrices=pruning_matrices)

    #Saliency Analysis
    s = saliency(nnet)
    indices_to_remove = base.indices_of_smallest_nonzero_k(s, remove)

    #Update Pruning Matrices
    pruning_matrices = base.weights_to_parameter_vector(pruning_matrices)
    pruning_matrices[indices_to_remove] = 0
    pruning_matrices = base.parameter_vector_to_weights(pruning_matrices,nnet['Gradients'])

    # Retrain after pruning
    nnet = train.gradient_descent(nnet, step_size=0.25, max_iterations=6000, pruning_matrices=pruning_matrices)
    return nnet

def iterative_prune(nnet, itr, remove):
    #Initialization
    weights = nnet['Weights']
    pruning_matrices = [np.ones(x.shape) for x in weights]

    #Initial Training
    nnet = train.gradient_descent(nnet, step_size=0.25, max_iterations=4000, pruning_matrices=pruning_matrices)

    for i in range(itr):
        #Saliency Analysis
        s = saliency(nnet)
        indices_to_remove = base.indices_of_smallest_nonzero_k(s,remove)

        #Update Pruning Matrices
        pruning_matrices = base.weights_to_parameter_vector(pruning_matrices)
        pruning_matrices[indices_to_remove] = 0
        pruning_matrices = base.parameter_vector_to_weights(pruning_matrices,nnet['Gradients'])

        # Retrain after pruning
        nnet = train.gradient_descent(nnet, step_size=0.25, max_iterations=4000, pruning_matrices=pruning_matrices)
    print(pruning_matrices)
    return nnet
def saliency(nnet):
    weights = nnet['Weights']
    vec_weights = [base.to_vector(w) for w in weights]
    vec_weights = np.vstack(vec_weights)
    hessian = hessian_matrix(nnet)
    inverse_hessian = np.linalg.inv(hessian)
    diagonals = np.array(np.diag(inverse_hessian))
    diagonals = diagonals.reshape((len(vec_weights),1))
    saliency_statistic = vec_weights**2 / (2 * abs(diagonals))
    return saliency_statistic

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
    return full_hessian



