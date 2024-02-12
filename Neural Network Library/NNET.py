import Base as base
import numpy as np
import ActivationFunctions as af
class neural_network:
    def __init__(self, layers, activation_functions, cost_function):

        # Input variables
        self.layers = layers
        self.weights = []
        self.augmented_states=[]
        self.augmented_weights = []
        self.states = []
        self.gradients = []

        # Cost function
        self.cost_function = cost_function
        self.costs = []

        # Activation functions and the derivative functions
        self.activation_functions = activation_functions
        activation_jacobian_names = [function.__name__ + "_derivative" for function in self.activation_functions]
        activation_hessian_names = [function.__name__ + "_second_derivative" for function in self.activation_functions]
        self.activation_jacobian_functions = [getattr(af, function) for function in activation_jacobian_names]
        self.activation_hessian_functions = [getattr(af, function) for function in activation_hessian_names]
        self.activation_jacobian_matrices = []
        self.activation_hessian_matrices = []

        #Adjoint variables
        self.lambdas = []
        self.thetas = []
        self.omegas = []

    def randomize_weights(self):
        self.weights = []
        for i in range(len(self.activation_functions)):
            self.weights.append(np.random.rand(self.layers[i]+1,self.layers[i+1]))

    def forward(self,data):
        self.states = [data.x]
        self.augmented_states = []
        self.activation_jacobian_matrices = []

        for i in range(len(self.activation_functions)):
            ones_column = np.ones((self.states[i].shape[0], 1))
            self.augmented_states.append(np.hstack((ones_column, self.states[i])))
            z = self.augmented_states[i] @ self.weights[i]
            self.states.append(self.activation_functions[i](z))
            self.activation_jacobian_matrices.append(np.diagflat(base.to_vector(self.activation_jacobian_functions[i](z))))

    def backward(self,data):
        n = self.states[-1].shape[0]
        λ = base.to_vector(self.states[-1] - data.y)
        self.lambdas = [λ]
        self.gradients = []
        self.augmented_weights = []

        for i in reversed(range(len(self.activation_functions))):
            p = self.layers[i+1]
            no_bias_weight = self.weights[i][1:,:]

            gradient = np.kron(np.eye(p),self.augmented_states[i]).T \
                       @ self.activation_jacobian_matrices[i] \
                       @ λ

            new_lambda = np.kron(no_bias_weight, np.eye(n)) \
                    @ self.activation_jacobian_matrices[i] \
                    @ λ

            gradient = base.to_matrix(gradient, self.weights[i].shape)
            self.gradients.append(gradient)
            self.lambdas.append(new_lambda)
            self.augmented_weights.append(no_bias_weight)
            λ = new_lambda

        self.gradients.reverse()
        self.lambdas.reverse()
        self.augmented_weights.reverse()

    def Av_Tensor(self, i):
        #Need to save (augmented weights)
        vector = self.thetas[i]
        n = self.augmented_states[0].shape[0]
        Av = np.kron(self.augmented_states[i], np.eye(n)) \
             @ np.diagflat(self.lambdas[i+1]) \
             @ self.activation_hessian_matrices[i] \
             @ np.kron(self.augmented_weights[i], np.eye(n)).T \
             @ vector
        return Av

    def Bv_Tensor(self, vectors, i):
        # Tensor-vector product for an x- and w- derivative of model equation
        n = self.augmented_states[0].shape[0]
        p = self.augmented_weights[i].shape[0]
        q = self.augmented_weights[i].shape[1]
        Bv = np.kron(self.lambdas[i + 1].T @ self.activation_jacobian_matrices[i], np.eye(n * p)) \
             @ K1v_Product(self.weights[i], n, vectors[i]) \
             + np.kron(self.augmented_weights[i], np.eye(n)) \
             @ np.diagflat(self.lambdas[i + 1]) \
             @ self.activation_hessian_matrices[i] \
             @ np.kron(np.eye(q), self.augmented_states[i]) \
             @ vectors[i]
        return Bv

    def Cv_Tensor(self, i):
        # Tensor-vector product for w- and x- derivatives of model equation
        vector = self.thetas[i]
        n = self.augmented_states[0].shape[0]
        p = self.augmented_weights[i].shape[0]
        q = self.augmented_weights[i].shape[1]
        v = np.kron(self.activation_jacobian_matrices[i] @ self.lambdas[i + 1], np.eye(n * p)) @ vector
        Cv = K2v_Product(self.weights[i], n, v) \
             + (np.kron(self.augmented_weights[i], np.eye(n))
                @ np.diagflat(self.lambdas[i + 1])
                @ self.activation_hessian_matrices[i]
                @ np.kron(np.eye(q), self.augmented_states[i])).T \
             @ vector
        return Cv

    def Dv_Tensor(self, vectors, i):
        # Tensor-vector product for two w-derivatives of model equation
        q = self.augmented_weights[i].shape[1]
        Dv = np.kron(np.eye(q), self.augmented_states[i].T) \
             @ np.diagflat(self.lambdas[i + 1]) \
             @ self.activation_hessian_matrices[i] \
             @ np.kron(np.eye(q), self.augmented_states[i]) \
             @ vectors[i]
        return Dv

    def update(self, step_size=0.05):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - step_size * self.gradients[i]

    def track_cost(self,df):
        predictions = self.states[-1]
        cost = self.cost_function(df.y,predictions)
        self.costs.append(cost)
    def train(self,df,max_iterations=5000, step_size=0.05):
        for i in range(max_iterations):
            self.forward(df)
            self.backward(df)
            self.track_cost(df)
            self.update(step_size)


def K1v_Product(weight, n, vector):
    # Tensor-vector product for eliminating Kronecker product from second derivative
    matrix = base.to_matrix(vector, weight.shape)
    matrix = matrix[1:, ]
    out = np.kron(matrix, np.eye(n))
    out = base.to_vector(out)
    return out

def K2v_Product(weight, n, vector):
    # Tensor-vector product for eliminating Kronecker product from second derivative
    p = weight.shape[0] - 1
    q = weight.shape[1]
    vector = base.to_matrix(vector, (n * p * n, q))
    P = np.eye(n * p)
    P = base.to_vector(P)
    P = base.to_matrix(P, (p, n * n * p))
    out = P @ vector
    zero = np.zeros((1, q))
    out = np.vstack((zero, out))
    out = base.to_vector(out)
    return out