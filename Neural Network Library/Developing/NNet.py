import numpy as np
from Math import *

class neural_network:
    def __init__(self, predictors, responses, layers, activations, cost_function):
        self.predictors = predictors
        self.responses = responses
        self.layers = layers
        self.activations = activations

        self.costs = [globals()[cost_function](self.predictors, self.responses)]
        self.weights = create_weights(self.layers)
        self.augmented_weights = [remove_bias(w) for w in self.weights]

        self.states = None
        self.augmented_states = None
        self.FOA_lambdas = None
        self.TLM_thetas = None
        self.SOA_omegas = None
        self.activation_jacobians = None
        self.activation_hessians = None
        self.gradients = None
        self.hessian_vector_products = None

    def forward(self):
        self.states = self.forward_model(self.predictors,update_network=True)
        self.augmented_states = [augment_design_matrix(x) for x in self.states]

    def predict(self, x):
        return self.forward_model(x, update_network=False)

    def forward_model(self,x,update_network=False):
        states = [x]
        for i in range(len(self.layers)-1):
            is_output_layer = False if i+1 < len(self.layers)-1 else True
            x = augment_design_matrix(x)
            y = x @ self.weights[i]
            y = globals()[self.activations[is_output_layer]](y)
            states.append(y)
            x = y
        return y if update_network is False else states



def augment_design_matrix(x):
    rows = x.shape[0]
    ones_column = np.ones((rows, 1))
    x = np.hstack((ones_column, x))
    return x

def create_weights(layers):
    weights = []
    l = len(layers)
    for i in range(l-1):
        next_layer = int(layers[i+1])
        current_layer = int(layers[i])
        weight = np.random.rand(current_layer+1,next_layer)*0.5
        weights.append(weight)
    return weights

def remove_bias(weight):
    return np.delete(weight, 0, axis=0)


