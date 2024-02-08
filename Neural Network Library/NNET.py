import Base as base
import numpy as np
import ActivationFunctions as af
import CostFunctions as cf
class neural_network:
    def __init__(self, layers, activation_functions, cost_function):

        # Input variables
        self.layers = layers
        self.weights = []

        # Cost function
        self.costs = cost_function

        # Activation functions and the derivative functions
        self.activation_functions = activation_functions
        activation_jacobian_names = [function.__name__ + "_derivative" for function in self.activation_functions]
        activation_hessian_names = [function.__name__ + "_second_derivative" for function in self.activation_functions]
        self.activation_jacobian_functions = [getattr(af, function) for function in activation_jacobian_names]
        self.activation_hessian_functions = [getattr(af, function) for function in activation_hessian_names]
        self.activation_jacobian_matrix = None
    def randomize_weights(self):
        self.weights = []
        for i in range(len(self.activation_functions)):
            self.weights.append(np.random.rand(self.layers[i]+1,self.layers[i+1]))

    def forward(self,data):
        ones_column = np.ones((data.training_x.shape[0], 1))
        self.states = [data.training_x]
        self.augmented_states = []
        self.activation_jacobian_matrices = []

        for i in range(len(self.activation_functions)):
            ones_column = np.ones((self.states[i].shape[0], 1))
            self.augmented_states.append(np.hstack((ones_column, self.states[i])))
            z = self.augmented_states[i] @ self.weights[i]
            self.states.append(self.activation_functions[i](z))
            self.activation_jacobian_matrices.append(np.diagflat(base.to_vector(self.activation_jacobian_functions[i](z))))

    def backward(self,data):
        n = self.states[-1]
        λ = base.to_vector(self.states[-1] - data.training_y)
        self.lambdas = [λ]
        self.gradients = []

        for i in reversed(range(len(self.activation_functions))):
            p = self.layers[i]
            z = base.to_vector(self.augmented_states[i] @ self.weights[i])
            jacobian = self.activation_jacobian_functions[i](z)
            no_bias_weight = self.weights[i][1:,:]
            print(np.kron(np.eye(p),self.augmented_states[i]).shape)
            print(self.activation_jacobian_matrices[i].shape)
            print(λ.shape)
            gradient = np.kron(np.eye(p),self.augmented_states[i]) \
                       @ self.activation_jacobian_matrices[i] \
                       @ λ

            new_λ = np.kron(no_bias_weight, np.eye(n)) \
                    @ self.activation_jacobian_matrices[i] \
                    @ λ

            λ = new_lambda



class data:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.training_x = None
        self.training_y = None
        self.validation_x = None
        self.validation_y = None

    def load_training_data(self,x,y):
        self.training_x = x
        self.training_y = y

    def load_validation_data(self,x,y):
        self.validation_x = x
        self.validation_y = y

    def test_train_split(self,train_percent=0.8, seed=None):
        rows = self.x.shape[0]
        indices = base.generate_random_indices(rows, random_seed=seed)
        split = int(np.round(rows * train_percent))
        train_indices = indices[0:split]
        test_indices = indices[split:]
        self.training_x = self.x[train_indices]
        self.training_y = self.y[train_indices]
        self.validation_x = self.x[test_indices]
        self.validation_y = self.y[test_indices]
        self.x = None
        self.y = None


"""The data set of predictor variables"""
x = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],
              [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]]).T


"""The data set of outcome variables"""
y = np.array([[1,1,1,1,1,0,0,0,0,0],
              [0,0,0,0,0,1,1,1,1,1]]).T

"""Define the neural network structure"""
np.random.seed(100)

df = data(x,y)
df.test_train_split(train_percent=0.8)
nnet = neural_network(layers=[2,2,3,2],
                      activation_functions = [af.sigmoid,af.sigmoid,af.sigmoid],
                      cost_function=cf.mean_squared_error)

nnet.randomize_weights()
nnet.forward(df)
nnet.backward(df)