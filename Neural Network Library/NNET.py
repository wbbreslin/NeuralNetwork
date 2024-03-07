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
        self.hvps = []
        self.predictions = None
        self.hessian_matrix = None
        self.gradient_vector = None

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

        self.randomize_weights()

    def randomize_weights(self):
        self.weights = []
        for i in range(len(self.activation_functions)):
            self.weights.append(0.5*np.random.rand(self.layers[i]+1,self.layers[i+1]))

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

    def predict(self,data):
        states = [data.x]
        augmented_states = []

        for i in range(len(self.activation_functions)):
            ones_column = np.ones((states[i].shape[0], 1))
            augmented_states.append(np.hstack((ones_column, states[i])))
            z = augmented_states[i] @ self.weights[i]
            states.append(self.activation_functions[i](z))

        self.predictions = states[-1]

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

    def backward_hyperparameter_derivative(self, data):
        n = self.states[-1].shape[0]
        self.dJ =[]
        for j in range(n):
            dλ = base.unit_matrix(j,n) @ (self.states[-1] - data.y)
            dλ = base.to_vector(dλ)
            current_dJ = []

            for i in reversed(range(len(self.activation_functions))):
                p = self.layers[i+1]
                no_bias_weight = self.weights[i][1:,:]

                gradient = dλ.T \
                           @ self.activation_jacobian_matrices[i] \
                           @ np.kron(np.eye(p),self.augmented_states[i])

                new_dλ = (dλ.T
                        @ self.activation_jacobian_matrices[i]
                        @ np.kron(no_bias_weight.T, np.eye(n))).T

                dλ = new_dλ
                current_dJ.append(gradient)
            current_dJ.reverse()
            current_dJ = np.hstack(current_dJ)
            self.dJ.append(current_dJ)
        self.dJ = np.vstack(self.dJ)

    def soa_forward(self, vectors):
        # Forward pass through the tangent-linear model
        theta = np.zeros(self.states[0].shape)
        theta = base.to_vector(theta)
        self.thetas = [theta]
        self.activation_hessian_matrices = []
        n = self.augmented_states[0].shape[0]

        for i in range(len(self.augmented_weights)):
            q = self.augmented_weights[i].shape[1]
            new_theta = self.activation_jacobian_matrices[i] \
                        @ np.kron(self.augmented_weights[i].T, np.eye(n)) \
                        @ self.thetas[i] \
                        + self.activation_jacobian_matrices[i] \
                        @ np.kron(np.eye(q), self.augmented_states[i]) \
                        @ vectors[i]
            self.thetas.append(new_theta)
            xw = self.augmented_states[i] @ self.weights[i]
            xw_vec = base.to_vector(xw)
            d2 = np.diagflat(self.activation_hessian_functions[i](xw_vec))
            self.activation_hessian_matrices.append(d2)

    def soa_backward(self, vectors):
        # Backward pass through the second-order adjoint model
        n = self.states[0].shape[0]
        omega = self.thetas[-1]
        self.omegas = [omega]
        self.hvps = []
        for i in reversed(range(len(self.augmented_weights))):
            q = self.weights[i].shape[1]
            gradient = np.kron(self.augmented_weights[i], np.eye(n)) @ self.activation_jacobian_matrices[i]
            Hv = np.kron(np.eye(q), self.augmented_states[i].T) @ self.activation_jacobian_matrices[i] \
                 @ omega \
                 + self.Dv_Tensor(vectors, i) \
                 + self.Cv_Tensor(i)
            new_omega = gradient @ omega + self.Av_Tensor(i) + self.Bv_Tensor(vectors, i)
            self.omegas.append(new_omega)
            dims = self.gradients[i].shape
            Hv = base.to_matrix(Hv, dims)
            self.hvps.append(Hv)
            omega = new_omega

        self.omegas.reverse()
        self.hvps.reverse()


    def Av_Tensor(self, i):
        vector = self.thetas[i]
        n = self.augmented_states[0].shape[0]
        Av = np.kron(self.augmented_weights[i], np.eye(n)) \
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

    def compute_gradient(self):
        self.gradient_vector = [[1]]
        for g in self.gradients:
            gradient = base.to_vector(g)
            self.gradient_vector = np.vstack((self.gradient_vector, gradient))
        self.gradient_vector = self.gradient_vector[1:]


    def compute_hessian(self):
        elements = [w.size for w in self.weights]
        partitions = np.append(0, np.cumsum(elements))
        dimensions = partitions[-1]
        full_hessian = np.zeros((dimensions, dimensions))

        for i in range(dimensions):
            vector = np.zeros((dimensions, 1))
            vectors = []
            vector[i] = 1
            for j in range(len(self.weights)):
                v = vector[partitions[j]:partitions[j + 1]]
                vectors.append(v)
            self.soa_forward(vectors)
            self.soa_backward(vectors)
            columns = []
            for k in range(len(self.weights)):
                hvp = base.to_vector(self.hvps[k])
                columns.append(hvp)
            column_hessian = columns[0]
            for c in range(len(columns) - 1):
                column_hessian = np.vstack((column_hessian, columns[c + 1]))
            full_hessian[:, i] = column_hessian[:, 0]
        self.hessian_matrix = full_hessian


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