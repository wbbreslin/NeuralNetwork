import FirstOrderModel as fom
import numpy as np


def gradient_descent(nnet,
                     step_size = 0.25,
                     max_iterations = 10**6,
                     pruning_matrices = []):
    iterations = 0

    while iterations < max_iterations:
        iterations = iterations + 1
        nnet = fom.first_order_model(nnet)
        for k in range(len(nnet['Gradients'])):
            if len(pruning_matrices)==0:
                nnet['Weights'][k] = nnet['Weights'][k] - step_size * nnet['Gradients'][k]
            else:
                nnet['Weights'][k] = nnet['Weights'][k] * pruning_matrices[k] - step_size * pruning_matrices[k] * nnet['Gradients'][k]

    return nnet

def stochastic_gradient(nnet,
                        subset_size,
                        step_size=0.25,
                        max_iterations = 10**6):

    n = nnet['Predictors'].shape[0]
    iterations = 0

    while iterations < max_iterations:
        iterations = iterations + 1
        random_indices = np.sort(np.random.choice(n, size=subset_size, replace=False))
        nnet['Pass Forward'] = nnet['Predictors'][random_indices, :]
        nnet['Outcomes_Subset'] = nnet['Outcomes'][random_indices, :]
        nnet = fom.first_order_model(nnet)
        for k in range(len(nnet['Gradients'])):
            nnet['Weights'][k] = nnet['Weights'][k] - step_size * nnet['Gradients'][k]

    return nnet

