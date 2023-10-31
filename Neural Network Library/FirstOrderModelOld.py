import numpy as np
import Base as base

#Update finite difference code, then delete
def first_order_model(x_predictors, y_outcomes, weights):
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
    states, augmented_states, first_derivatives = forward_pass(x_predictors,
                                                               weights)

    Lambdas, gradients = backward_pass(states,
                                       y_outcomes,
                                       first_derivatives,
                                       weights)
    y_predictions = states[-1]

    return (y_predictions,
            gradients,
            Lambdas)

def forward_pass(x_predictors, weights):
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
    states = [x_predictors]
    augmented_states = []
    first_derivatives = []

    for i in range(len(weights)):
        z, A, B = base.augment_predictor(x_predictors)
        augmented_states.append(z)
        xw = z @ weights[i]
        x_predictors = base.sigmoid(xw)
        d1 = base.sigmoid_derivative(xw)
        first_derivatives.append(d1)
        states.append(x_predictors)

    return (states,
            augmented_states,
            first_derivatives)

def backward_pass(states, y_outcomes, first_derivatives, weights):
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
    terminal_state = states[-1]
    n = terminal_state.shape[0]
    Lambda, dims = base.to_vector(terminal_state - y_outcomes)
    p = dims[1]
    Lambdas = [Lambda]
    gradients = []
    dimensions = [p]

    for i in reversed(range(len(weights))):
        Z, A, B = base.augment_predictor(states[i])
        gradient = np.kron(np.eye(p),Z).T @ first_derivatives[i] @ Lambda
        new_Lambda = np.kron((A @ weights[i]).T,np.eye(n)).T @ first_derivatives[i] @ Lambda
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
        Lambdas[i] = base.to_matrix(Lambdas[i],(n,dimensions[i-1]))

    return Lambdas, gradients