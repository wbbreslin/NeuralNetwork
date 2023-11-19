import numpy as np
from Math import *

class neural_network:
    def __init__(self, predictors, responses, layers, activations, cost_function):
        """
        Inputs
        -----------------------------------------------------
        predictors      - a numpy array of (X) predictor variables in design matrix layout
        responses       - a numpy array of (Y) response variables
        layers          - a list of the number of neurons in each layer of the network
        activations     - a list of strings containing function names for activation functions
        cost_function   - a string for the name of the cost functional to be used
        """

        # Variables defined directly through class inputs
        self.predictors = predictors
        self.responses = responses
        self.layers = layers
        self.activations = activations

        # Variables defined by external functions
        self.weights = create_weights(self.layers)
        self.costs = [globals()[cost_function](self.predictors, self.responses)]

        # Variables that will be created in forward or backward passes
        self.states = None
        self.FOA = None
        self.TLM = None
        self.SOA = None
        self.activation_jacobians = None
        self.activation_hessians = None

        # Important output variables
        self.gradients = None
        self.hessian_vector_products = None

    def forward(self,x):
        return self.predict(x,update_network=True)
    def predict(self,x,update_network=False):
        states = [x]
        for i in range(len(self.layers)-1):
            is_output_layer = False if i+1 < len(self.layers)-1 else True
            x = augment_design_matrix(x)
            y = x @ self.weights[i]
            y = globals()[self.activations[is_output_layer]](y)
            states.append(y)
            x = y
        return y if update_network is False else states



def augment_design_matrix(x_predictors):
    dimensions = x_predictors.shape
    A1 = np.zeros((dimensions[1],1))
    A2 = np.eye(dimensions[1])
    A = np.hstack((A1,A2))
    B1 = np.ones((dimensions[0],1))
    B2 = np.zeros(dimensions)
    B = np.hstack((B1,B2))
    Z = x_predictors @ A + B
    return Z
def create_weights(layers):
    weights = []
    l = len(layers)
    for i in range(l-1):
        next_layer = int(layers[i+1])
        current_layer = int(layers[i])
        weight = np.random.rand(current_layer+1,next_layer)*0.5
        weights.append(weight)
    return weights




x = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],
              [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]]).T

"""The data set of outcome variables"""
y= np.array([[1,1,1,1,1,0,0,0,0,0],
             [0,0,0,0,0,1,1,1,1,1]]).T

layers = [2,2,3,2]
activations = ['relu', 'sigmoid']
cost = 'MSE'
nnet = neural_network(x,y,layers,activations,cost)
print(nnet.costs)

predictions = nnet.predict(x)
fpass = nnet.forward(x)
print(predictions)
print(fpass)