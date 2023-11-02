import numpy as np
import Base as base

def first_order_model(nnet):
    """
    Description
    --------------------
    This is a wrapper function combining the forward pass and backward passes into one function.

    Inputs
    --------------------
    x_predictors        : A matrix (n x p) containing n data points in p variables
    y_outcomes          : A matrix (n x q) containing n data points in q variables
    weights             : List of the weight parameter matrices for the NN

    Outputs
    --------------------
    states              : List of states (x) at each layer of the NN
    gradients           : List of gradients for each weight matrix
    Lambdas             : List of adjoint variable states at each layer of the NN
    """
    nnet = forward_pass(nnet)
    nnet = backward_pass(nnet)

    return nnet

def forward_pass(nnet):
    """
    Description
    --------------------
    Neural Network model dynamics (Forward Pass)

    Inputs
    --------------------
    x_predictors        : A matrix (n x p) containing n data points in p variables
    weights             : List of the weight parameter matrices for the NN

    Outputs
    --------------------
    states              : List of states (x) at each layer of the NN
    first_derivatives   : List of Sigmoid first derivative matrices for each NN layer
    """
    states = [nnet['Predictors']]
    augmented_states = []
    augmented_weights = []
    first_derivatives = []
    x_predictors = nnet['Predictors']
    weights = nnet['Weights']
    for i in range(len(nnet['Weights'])):
        z, A, B = base.augment_predictor(x_predictors)
        augmented_states.append(z)
        xw = z @ weights[i]
        x_predictors = base.sigmoid(xw)
        xw_vec, dims = base.to_vector(xw)
        d1 = np.diagflat(base.sigmoid_derivative(xw_vec))
        first_derivatives.append(d1)
        states.append(x_predictors)
        aug_weight = A @ weights[i]
        augmented_weights.append(aug_weight)

    output = {'States': states,
              'Augmented_States': augmented_states,
              'Augmented_Weights': augmented_weights,
              'First_Derivatives': first_derivatives}

    nnet.update(output)

    return nnet

def backward_pass(nnet):
    """
    Description
    --------------------
    First-order adjoint model (Backward Pass)

    Inputs
    --------------------
    states              : List of states (x) at each layer of the NN
    y_outcomes          : A matrix (n x q) containing n data points in q variables
    first_derivatives   : List of Sigmoid first derivative matrices for each NN layer
    weights             : List of the weight parameter matrices for the NN


    Outputs
    --------------------
    Lambdas             : List of adjoint variable states at each layer of the NN
    gradients           : List of gradients for each weight matrix
    """
    states = nnet['States']
    terminal_state = states[-1]
    y_outcomes = nnet['Outcomes']
    first_derivatives = nnet['First_Derivatives']
    weights = nnet['Weights']

    n = terminal_state.shape[0]
    Lambda, dims = base.to_vector(terminal_state - y_outcomes)
    p = dims[1]
    Lambdas = [Lambda]
    gradients = []
    dimensions = [p]

    for i in reversed(range(len(weights))):
        Z, A, B = base.augment_predictor(states[i])

        gradient = np.kron(np.eye(p),Z.T) \
                   @ first_derivatives[i] \
                   @ Lambda

        new_Lambda = np.kron((A @ weights[i]),np.eye(n))\
                     @ first_derivatives[i] \
                     @ Lambda

        gradients.append(gradient)
        Lambdas.append(new_Lambda)
        Lambda = new_Lambda
        p = int(Lambda.shape[0]/n)
        dimensions.append(p)

    gradients.reverse()
    Lambdas.reverse()

    for i in range(len(gradients)):
        dims = weights[i].shape
        gradients[i] = base.to_matrix(gradients[i],dims)
        #Lambdas[i] = base.to_matrix(Lambdas[i],(n,dimensions[i-1]))

    output = {'Lambdas': Lambdas,
              'Gradients': gradients}

    nnet.update(output)

    return nnet