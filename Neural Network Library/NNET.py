import Base as base
import numpy as np
import ActivationFunctions as af
import CostFunctions as cf
class neural_network:
    def __init__(self, layers, activation_functions, cost_function):

        # Training Data
        x_training = None
        y_training = None

        # Validation Data
        x_validation = None
        y_training = None

        # Cost function
        self.costs = cost_function

        # Activation functions and the derivative functions
        self.activation_functions = activation_functions
        activation_jacobian_names = [function.__name__ + "_derivative" for function in self.activation_functions]
        activation_hessian_names = [function.__name__ + "_second_derivative" for function in self.activation_functions]
        self.activation_jacobian_functions = [getattr(af, function) for function in activation_jacobian_names]
        self.activation_hessian_functions = [getattr(af, function) for function in activation_hessian_names]

    def load_training_data(self,x_training,y_training):
        self.x_training = x_training
        self.y_training = y_training



"""The data set of predictor variables"""
x_training = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],
                        [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]]).T
"""The data set of outcome variables"""
y_training = np.array([[1,1,1,1,1,0,0,0,0,0],
                      [0,0,0,0,0,1,1,1,1,1]]).T

"""Define the neural network structure"""
np.random.seed(100)

nnet = neural_network(layers=[2,4,4,2],
                      activation_functions = [af.sigmoid,af.sigmoid,af.sigmoid],
                      cost_function=cf.mean_squared_error)

nnet.load_training_data(x_training, y_training)

#print(nnet.activation_functions)
#print(nnet.activation_jacobian_functions)
#print(nnet.activation_hessian_functions)

print(nnet.x_training)
