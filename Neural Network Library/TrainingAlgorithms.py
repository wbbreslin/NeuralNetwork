import FirstOrderModel


def gradient_descent(nnet,
                     step_size = 0.25,
                     max_iterations = 10**6):
    """
    Description
    --------------------
    The gradient descent algorithm.

    Inputs
    --------------------
    x_predictors        : A matrix (n x p) containing n data points in p variables
    y_outcomes          : A matrix (n x q) containing n data points in q variables
    weights             : List of the weight parameter matrices for the NN
    gradient_function   : A function that computes the gradient
    step_size           : The starting step size to scale the gradient, default is 0.25
    tolerance           : The error tolerance to terminate the algorithm, default is 10**-4
    max_iterations      : List of the weight parameter matrices for the NN
    backtracking        : Boolean value to enable backtracking line search, default will be True

    Outputs
    --------------------
    y_predictions       : A matrix (n x q) containing n data points in q variables
    gradients           : List of gradients for each weight matrix
    """
    iterations = 0

    while iterations < max_iterations:
        iterations = iterations + 1
        nnet = FirstOrderModel.first_order_model(nnet)
        for k in range(len(nnet['Gradients'])):
            nnet['Weights'][k] = nnet['Weights'][k] - step_size * nnet['Gradients'][k]

    return nnet
