import numpy as np

def TLM_model(states,
              weights,
              first_derivatives,
              second_derivatives,
              Lambdas,
              vectors):
    """
    Tangent Linear Model (Forward Pass 2)

    Inputs
    --------------------
    states              : List of states (x) at each layer of the NN
    weights             : A list of the weight parameter matrices for the NN
    first_derivatives   : List of Sigmoid first-derivative matrices for each NN layer
    second_derivatives  : List of Sigmoid second-derivative matrices for each NN layer
    Lambdas             : List of adjoint model states at each layer of the NN
    vectors             : A list of vectors for the Hessian-vector product calculation


    Outputs
    --------------------
    Xis                 : List of tangent linear model states at each layer of the NN
    A_matrices          : List of A matrices used for Hessian-vector product calculation
    B_tensors           : List of B tensors used for Hessian-vector product calculation
    C_tensors           : List of C tensors used for Hessian-vector product calculation
    D_tensors           : List of D tensors used for Hessian-vector product calculation
    """

    Xi = np.zeros(states[0].shape)
    Xis = [Xi]
    A_matrices = []
    B_tensors = []
    C_tensors = []
    D_tensors = []

    for i in range(len(weights)):
        LHS = first_derivatives[i] @ weights[i] @ Xi
        RHS = np.kron(states[i].T, first_derivatives[i]) @ vectors[i]
        new_Xi = LHS + RHS
        A = A_matrix(weights[i], Lambdas[i + 1], second_derivatives[i])
        B = B_tensor(states[i], weights[i], Lambdas[i + 1], first_derivatives[i], second_derivatives[i])
        C = C_tensor(states[i], weights[i], Lambdas[i + 1], first_derivatives[i], second_derivatives[i])
        D = D_tensor(states[i], Lambdas[i + 1], second_derivatives[i])
        A_matrices.append(A)
        B_tensors.append(B)
        C_tensors.append(C)
        D_tensors.append(D)
        Xis.append(new_Xi)
        Xi = new_Xi

    return (Xis,
            A_matrices,
            B_tensors,
            C_tensors,
            D_tensors)


def SOA_model(states,
              weights,
              first_derivatives,
              Xis,
              vectors,
              A_matrices,
              B_tensors,
              C_tensors,
              D_tensors):
    """
    Tangent Linear Model (Forward Pass 2)

    Inputs
    --------------------
    states              : List of states (x) at each layer of the NN
    weights             : A list of the weight parameter matrices for the NN
    first_derivatives   : List of Sigmoid first derivative matrices for each NN layer
    Xis                 : List of tangent-linear model states at each layer of the NN
    vectors             : A list of vectors for the Hessian-vector product calculation
    A_matrices          : List of A matrices used for Hessian-vector product calculation
    B_tensors           : List of B tensors used for Hessian-vector product calculation
    C_tensors           : List of C tensors used for Hessian-vector product calculation
    D_tensors           : List of D tensors used for Hessian-vector product calculation

    Outputs
    --------------------
    Etas                : List of second-order adjoint model states at each layer of the NN
    HVPs                : List of Hessian-vector products for each NN layer
    """

    Eta = Xis[-1]
    Etas = [Eta]
    HVPs = []
    for i in reversed(range(len(weights))):
        left = weights[i].T @ first_derivatives[i] @ Eta
        middle = A_matrices[i] @ Xis[i]
        right = C_tensors[i] @ vectors[i]
        new_Eta = left + middle + right
        HVP = Hessian_Vector_Product(states[i],
                                     first_derivatives[i],
                                     Eta,
                                     Xis[i],
                                     B_tensors[i],
                                     D_tensors[i],
                                     vectors[i])
        HVP = HVP.reshape(weights[i].shape,
                          order='F')
        Etas.append(new_Eta)
        HVPs.append(HVP)
        Eta = new_Eta

    Etas.reverse()
    HVPs.reverse()
    return Etas, HVPs

def A_matrix(weight, Lambda, second_derivative):
    """
    Evaluation of Matrix (A) for the Hessian-vector Product
    
    Inputs
    --------------------
    weight               : A Weight matrix
    Lambda               : A Lambda vector (adjoint variable)
    second_derivative    : A Sigmoid second-derivative matrix

    Outputs
    --------------------
    A                    : The (n x n) matrix A
    """

    Lambda_matrix = np.diagflat(Lambda)
    A = weight.T @ Lambda_matrix @ second_derivative @ weight
    return A
    
def B_tensor(state, weight, Lambda, first_derivative, second_derivative):
    """
    Evaluation of Tensor (B) for the Hessian-vector Product
    
    Inputs
    --------------------
    state                : A single data point for the predictor variable (n-dimensional vector)
    weight               : An individual Weight matrix
    Lambda               : An individual lambda vector (adjoint variable)
    first_derivative     : A Sigmoid first-derivative matrix
    second_derivative    : A Sigmoid second-derivative matrix

    Outputs
    --------------------
    B                    : The (nm x n) tensor B
    """

    Lambda_matrix = np.diagflat(Lambda)
    Identity = np.eye(state.shape[0])
    LHS = np.kron(Lambda.T @ first_derivative, Identity)
    RHS = np.kron(state.T, weight.T @ Lambda_matrix @ second_derivative)
    B = (LHS+RHS).T
    return B

def C_tensor(state, weight, Lambda, first_derivative, second_derivative):
    """
    Evaluation of Tensor (C) for the Hessian-vector Product
    
    Inputs
    --------------------
    state                : A single data point for the predictor variable (n-dimensional vector)
    weight               : An individual Weight matrix
    Lambda               : An individual lambda vector (adjoint variable)
    first_derivative     : A Sigmoid first-derivative matrix
    second_derivative    : A Sigmoid second-derivative matrix

    Outputs
    --------------------
    C                    : The (n x nm) tensor C
    """

    Lambda_matrix = np.diagflat(Lambda)
    Identity = np.eye(state.shape[0])
    LHS = np.kron(state, Lambda_matrix @ second_derivative @ weight)
    RHS = np.kron(Identity, first_derivative @ Lambda)
    C = (LHS+RHS).T
    return C

def D_tensor(state, Lambda, second_derivative):
    """
    Evaluation of Tensor (D) for the Hessian-vector Product
    
    Inputs
    --------------------
    state                : A single data point for the predictor variable (n-dimensional vector)
    Lambda               : An individual lambda vector (adjoint variable)
    second_derivative    : A Sigmoid second-derivative matrix

    Outputs
    --------------------
    D                    : The (nm x nm) dimensional tensor D
    """

    Lambda_matrix = np.diagflat(Lambda)
    D = np.kron(np.outer(state,state), Lambda_matrix @ second_derivative)
    return D

def Hessian_Vector_Product(x, first_derivative, Eta, Xi, B_tensor, D_tensor, vector):
    HVP = np.kron(first_derivative, x) @ Eta + B_tensor @ Xi + D_tensor @ vector
    return HVP
