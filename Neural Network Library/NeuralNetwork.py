import Base as base
import numpy as np
import ActivationFunctions as af
class neural_network:
    def __init__(self, layers, activation_functions, cost_function,
                 validation_function=None, regularization = 0):

        # Activation functions and the derivative functions
        self.activation_functions = activation_functions
        activation_jacobian_names = [function.__name__ + "_derivative" for function in self.activation_functions]
        activation_hessian_names = [function.__name__ + "_second_derivative" for function in self.activation_functions]
        self.activation_jacobian_functions = [getattr(af, function) for function in activation_jacobian_names]
        self.activation_hessian_functions = [getattr(af, function) for function in activation_hessian_names]
        self.activation_jacobian_matrices = []
        self.activation_hessian_matrices = []

        # Cost and Validation function
        self.cost_function = cost_function
        self.costs = []
        self.validation_function = validation_function
        self.regularization = regularization

        #ADAM
        self.momentum = [0] * len(layers)
        self.velocity = [0] * len(layers)
        self.iter = 0

        # Initialize weights
        self.layers = layers
        self.weights = []
        for i in range(len(self.activation_functions)):
            self.weights.append(0.5*np.random.rand(self.layers[i+1],self.layers[i]+1))

        # FOM variables
        self.states = []
        self.augmented_states=[]
        self.augmented_weights = []
        self.lambdas = []
        self.gradients = []
        self.gradient_vector = None
        self.predictions = None

        #SOM variables
        self.thetas = []
        self.omegas = []
        self.hvps = []
        self.hessian_matrix = None
        self.dSW = None

        #Validation variables

    def forward(self,data):
        self.states = [data.x]
        self.augmented_states = []
        self.activation_jacobian_matrices = []

        for i in range(len(self.activation_functions)):
            ones_row = np.ones((1,self.states[i].shape[1]))
            self.augmented_states.append(np.vstack((ones_row, self.states[i])))
            z = self.weights[i] @ self.augmented_states[i]
            self.states.append(self.activation_functions[i](z))
            self.activation_jacobian_matrices.append(self.activation_jacobian_functions[i](z))


    def predict(self,data):
        states = [data.x]
        augmented_states = []

        for i in range(len(self.activation_functions)):
            ones_row = np.ones((1,self.states[i].shape[1]))
            augmented_states.append(np.vstack((ones_row, self.states[i])))
            z = self.weights[i] @ self.augmented_states[i]
            states.append(self.activation_functions[i](z))

        data.predictions = states[-1]

    def backward(self,data):
        n = self.states[-1].shape[1]
        adjoint_mat = (data.s * (self.states[-1] - data.y).T).T
        adjoint_vec = base.to_vector(adjoint_mat)
        self.lambdas = [adjoint_vec]
        self.gradients = []
        self.augmented_weights = []

        for i in reversed(range(len(self.activation_functions))):
            p = self.layers[i]
            q = self.layers[i+1]

            #Gradient calculation
            f_grad_w = x_kronecker_identity_times_block_matrix(self.augmented_states[i],
                                                               self.activation_jacobian_matrices[i])
            gradient = f_grad_w @ adjoint_vec

            #Adjoint calculation
            no_bias_weight = self.weights[i][:,1:]
            f_grad_x = no_bias_weight.T @ self.activation_jacobian_matrices[i]
            new_lambda = base.columnwise_tensor_matrix_product(f_grad_x, adjoint_mat)
            new_lambda = new_lambda.reshape(n*p,1)

            gradient = base.to_matrix(gradient, self.weights[i].shape)
            remove_bias = np.ones(gradient.shape)
            remove_bias[:,0] = 0
            gradient = gradient + self.weights[i] * self.regularization

            self.gradients.append(gradient)
            self.lambdas.append(new_lambda)
            self.augmented_weights.append(no_bias_weight)
            adjoint_vec = new_lambda
            adjoint_mat = base.to_matrix(new_lambda,(p,n))

        self.gradients.reverse()
        self.lambdas.reverse()
        self.augmented_weights.reverse()

    def backward_hyperparameter_derivative(self, data):
        n = self.states[-1].shape[1]
        self.dSW =[]
        for j in range(n):
            dλ_mat = (self.states[-1] - data.y) @ base.unit_matrix(j,n)
            dλ_vec = base.to_vector(dλ_mat)
            current_dSW = []

            for i in reversed(range(len(self.activation_functions))):
                p = self.layers[i+1]+1

                #Lambda derivative calculation
                no_bias_weight = self.weights[i][:,1:]
                f_grad_x = no_bias_weight.T @ self.activation_jacobian_matrices[i]
                new_lambda_derivative = base.columnwise_tensor_matrix_product(f_grad_x, dλ_mat)
                a = new_lambda_derivative.shape[0]
                b = new_lambda_derivative.shape[1]
                new_lambda_derivative = new_lambda_derivative.reshape(a*b, 1)

                # Gradient calculation
                f_grad_w = x_kronecker_identity_times_block_matrix(self.augmented_states[i],
                                                                   self.activation_jacobian_matrices[i])
                gradient = f_grad_w @ dλ_vec

                #Update and store values
                dλ_vec = new_lambda_derivative
                dλ_mat = base.to_matrix(new_lambda_derivative,(b,a))
                current_dSW.append(gradient)

            current_dSW.reverse()
            current_dSW = np.vstack(current_dSW)
            self.dSW.append(current_dSW)
        self.dSW = np.hstack(self.dSW)
        self.dSW = self.dSW.T

    def soa_forward(self, vectors):
        # Forward pass through the tangent-linear model
        theta_mat = np.zeros(self.states[0].shape)
        theta_vec = base.to_vector(theta_mat)
        self.thetas = [theta_vec]
        self.activation_hessian_matrices = []
        n = self.augmented_states[0].shape[1]

        for i in range(len(self.augmented_weights)):
            p = self.layers[i]
            q = self.layers[i + 1]

            f_grad_w = x_kronecker_identity_times_block_matrix(self.augmented_states[i],
                                                               self.activation_jacobian_matrices[i])

            f_grad_x = self.augmented_weights[i].T @ self.activation_jacobian_matrices[i]
            f_grad_x_transposed = np.transpose(f_grad_x, (0, 2, 1))

            #Updating Theta
            vec = base.to_vector(vectors[i])
            theta_mat = base.to_matrix(self.thetas[i],self.states[i].shape)
            theta_product = base.columnwise_tensor_matrix_product(f_grad_x_transposed, theta_mat)
            theta_product = theta_product.reshape(n*q, 1)
            theta_product = np.vstack(theta_product)
            new_theta = theta_product + f_grad_w.T @ vec

            self.thetas.append(new_theta)
            wx =  self.weights[i] @ self.augmented_states[i]
            d2 = self.activation_hessian_functions[i](wx)
            self.activation_hessian_matrices.append(d2)

    def soa_backward(self, vectors):
        # Backward pass through the second-order adjoint model
        n = self.states[0].shape[0]
        omega = self.thetas[-1]
        self.omegas = [omega]
        self.hvps = []
        for i in reversed(range(len(self.augmented_weights))):
            q = self.weights[i].shape[1]

            f_grad_x = self.augmented_weights[i].T @ self.activation_jacobian_matrices[i]
            f_grad_w = x_kronecker_identity_times_block_matrix(self.augmented_states[i],
                                                               self.activation_jacobian_matrices[i])

            Hv = f_grad_w @ omega + self.Dv_Tensor(vectors, i) + self.Cv_Tensor(i)
            new_omega = block_diag_times_vector(f_grad_x,omega) + self.Av_Tensor(i) + self.Bv_Tensor(vectors, i)
            self.omegas.append(new_omega)
            dims = self.gradients[i].shape
            Hv = base.to_matrix(Hv, dims)
            self.hvps.append(Hv)
            omega = new_omega


        self.omegas.reverse()
        self.hvps.reverse()

    def Av_Tensor(self,i):
        # Finished
        vector = self.thetas[i]
        n = self.augmented_states[0].shape[1]
        p = self.states[i].shape[0]
        w = self.augmented_weights[i]
        Av = lambda_kronecker_times_hessian(self.lambdas[i+1],
                                              self.activation_hessian_matrices[i])
        Av = w.T @ Av @ w
        matrix = base.to_matrix(vector, self.states[i].shape)
        Av = base.columnwise_tensor_matrix_product(Av, matrix)
        Av = Av.reshape(n*p,1)
        return Av

    def Bv_Tensor(self,vectors,i):
        n = self.augmented_states[0].shape[1]
        p = self.states[i].shape[0]
        q = self.states[i + 1].shape[0]
        Bv1 = lambda_kronecker_times_hessian(self.lambdas[i + 1],
                                             self.activation_hessian_matrices[i])
        Bv1 = x_kronecker_identity_times_block_matrix(self.augmented_states[i],
                                                      Bv1)
        Bv1 = Bv1 @ np.kron(np.eye(n), self.augmented_weights[i])
        Bv1 = Bv1.T @ vectors[i]

        adjoint_matrix = base.to_matrix(self.lambdas[i + 1], self.states[i + 1].shape)
        Bv2 = base.columnwise_tensor_matrix_product(self.activation_jacobian_matrices[i],
                                                    adjoint_matrix)
        Bv2 = np.kron(Bv2.reshape(n*q,1), np.eye((p+1)*q))
        Bv2 = Bv2 @ vectors[i]
        Bv2 = K2v_Product(Bv2,p,q,n)
        return Bv1 + Bv2

    def Cv_Tensor(self,i):
        vector = self.thetas[i]
        n = self.augmented_states[0].shape[1]
        p = self.states[i].shape[0]
        q = self.states[i+1].shape[0]
        Cv1 = lambda_kronecker_times_hessian(self.lambdas[i+1],
                                            self.activation_hessian_matrices[i])
        Cv1 = x_kronecker_identity_times_block_matrix(self.augmented_states[i],
                                                     Cv1)
        Cv1 = Cv1 @ np.kron(np.eye(n), self.augmented_weights[i]) @ vector

        adjoint_matrix = base.to_matrix(self.lambdas[i+1],self.states[i+1].shape)
        Cv2 = base.columnwise_tensor_matrix_product(self.activation_jacobian_matrices[i],
                                                    adjoint_matrix)

        Cv2 = np.kron(Cv2.reshape(n*q,1).T, np.eye((p+1)*q))
        Cv2 = Cv2 @ Kv(vector, (p,n),q)

        return Cv1 + Cv2

    def Dv_Tensor(self,vectors,i):
        # Finished
        q = self.states[i+1].shape[0]
        Dv = lambda_kronecker_times_hessian(self.lambdas[i+1],
                                            self.activation_hessian_matrices[i])
        Dv = x_kronecker_identity_times_block_matrix(self.augmented_states[i],
                                                     Dv)
        Dv = Dv @ np.kron(self.augmented_states[i].T, np.eye(q)) @ vectors[i]
        return Dv

    def update(self, step_size=0.05, decay=1):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - step_size * self.gradients[i]




    def track_cost(self,df):
        predictions = self.states[-1]
        cost = self.cost_function(df.y,predictions,df.s)
        #reg = [np.linalg.norm(w)**2 for w in self.augmented_weights]
        reg = [np.linalg.norm(w) ** 2 for w in self.weights]
        reg = np.sum(reg) / 2 * self.regularization
        cost = cost + reg
        self.costs.append(cost)

    def compute_cost(self,df,append=False):
        predictions = df.predictions
        cost = self.cost_function(df.y,predictions,df.s)
        #reg = [np.linalg.norm(w)**2 for w in self.augmented_weights]
        reg = [np.linalg.norm(w) ** 2 for w in self.weights]
        reg = np.sum(reg) / 2 * self.regularization
        cost = cost + reg
        if append == True:
            self.costs.append(cost)
        else:
            return(cost)

    def train(self,df,max_iterations=5000, step_size=0.05, decay=1):
        for i in range(max_iterations):
            self.forward(df)
            self.predict(df)
            self.compute_cost(df,append=True)
            self.backward(df)
            #self.update_backtracking(df,step_size)
            self.update(step_size)
            step_size = step_size * decay
            if i % 1000 == 0:
                print(step_size)

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
        reg_matrix = np.eye(dimensions) * self.regularization
        self.hessian_matrix = full_hessian + reg_matrix #need to actually put this in Hvp code


def K1v_Product(matrix, identity_dims, vector):
    #Need to fix this for new setup
    # Tensor-vector product for eliminating Kronecker product from second derivative
    vec_matrix = base.to_matrix(vector, matrix.shape)
    out = np.kron(vec_matrix, np.eye(identity_dims))
    out = base.to_vector(out)
    return out

'''This ended up the same code as before, just with fewer assumptions about the matrix'''
def Kv(vector,shape,dim):
    matrix = base.to_matrix(vector,shape)
    zero = np.zeros((1, shape[1]))
    matrix = np.vstack((zero, matrix))
    y = np.kron(matrix, np.eye(dim))
    y = base.to_vector(y)
    return y

def K2v_Product_old(weight, n, vector):
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

def K2v_Product(vector, p, q, n):
    # Tensor-vector product for eliminating Kronecker product from second derivative
    vector = base.to_matrix(vector, (q * (p+1) * q, n))
    I = np.eye(q * (p+1))
    I = base.to_vector(I)
    I = base.to_matrix(I, (p+1, q * q * (p+1)))
    out = I @ vector
    out = out[1:,:]
    out = base.to_vector(out)
    return out

def lambda_kronecker_times_hessian(lambda_vec, hessian_tensor):
    n = hessian_tensor.shape[0]
    q = hessian_tensor.shape[1]
    lambda_vec = lambda_vec.reshape(n, q, 1, 1)
    product = lambda_vec * hessian_tensor
    sum_product = np.sum(product,axis=1)
    return sum_product

def x_kronecker_identity_times_block_matrix(matrix,jacobian):
    # Compute (XxI)J, returns matrix
    # Jacobian must be a tensor
    p = matrix.shape[0]-1
    n = matrix.shape[1]
    q = jacobian.shape[1]
    matrix_reshaped = matrix[:, :, None, None]  # Convert to tensor
    product = matrix_reshaped * jacobian  # Broadcast multiplication
    product = product.transpose(0, 2, 1, 3).reshape(q * (p + 1), n * q)  # Reshape to matrix representation
    return product

def block_diag_times_vector(tensor,vector):
    n = tensor.shape[0]
    a = tensor.shape[1]
    b = tensor.shape[2]
    reshaped_vector = vector.reshape(n,1,b)
    product = tensor * reshaped_vector
    product = np.sum(product, axis=2)
    product = product.reshape(n*a,1)
    return product
