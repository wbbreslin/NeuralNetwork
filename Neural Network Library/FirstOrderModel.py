import numpy as np
import Base as base

def first_order_model(nnet):
    nnet = forward_pass(nnet)
    nnet = track_cost(nnet)
    nnet = backward_pass(nnet)
    return nnet

def forward_pass(nnet):
    states = [nnet['Pass Forward']]
    augmented_states = []
    augmented_weights = []
    first_derivatives = []
    x_predictors = nnet['Pass Forward']
    weights = nnet['Weights']
    for i in range(len(nnet['Weights'])):
        z, A, B = base.augment_predictor(x_predictors)
        augmented_states.append(z)
        xw = z @ weights[i]
        x_predictors = base.sigmoid(xw)
        xw_vec = base.to_vector(xw)
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

def track_cost(nnet):
    y_predictions = nnet['States'][-1]
    y_outcomes = nnet['Outcomes_Subset']
    cost = base.mean_squared_error(y_predictions, y_outcomes)
    if 'Cost' in nnet:
        nnet['Cost'].append(cost)
    else:
        output = {'Cost': [cost]}
        nnet.update(output)

    return nnet

def backward_pass(nnet):
    states = nnet['States']
    terminal_state = states[-1]
    y_outcomes = nnet['Outcomes_Subset']
    first_derivatives = nnet['First_Derivatives']
    weights = nnet['Weights']

    n = terminal_state.shape[0]
    Lambda = base.to_vector(terminal_state - y_outcomes)
    p = terminal_state.shape[1]
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