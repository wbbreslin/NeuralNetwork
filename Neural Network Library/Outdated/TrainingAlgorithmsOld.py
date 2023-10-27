def gradient_descent(x_predictors,
                     y_outcomes,
                      weights,
                      gradient_function,
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
        y_predictions, gradients, Lambdas = gradient_function(x_predictors,y_outcomes,weights)
        for k in range(len(gradients)):
            weights[k] = weights[k] - step_size * gradients[k]

    return y_predictions, weights, gradients