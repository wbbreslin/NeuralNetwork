import numpy as np
import Base as base


def forward_pass(nnet, vectors):
    theta = np.zeros(nnet['Predictors'].shape)
    theta, dim = base.to_vector(theta)
    thetas = [theta]
    second_derivatives = []
    n = nnet['Augmented_States'][0].shape[0]

    for i in range(len(nnet['Augmented_Weights'])):
        q = nnet['Augmented_Weights'][i].shape[1]
        new_theta = nnet['First_Derivatives'][i] \
                    @ np.kron(nnet['Augmented_Weights'][i].T, np.eye(n)) \
                    @ thetas[i] \
                    + nnet['First_Derivatives'][i] \
                    @ np.kron(np.eye(q), nnet['Augmented_States'][i]) \
                    @ vectors[i]
        thetas.append(new_theta)
        xw = nnet['Augmented_States'][i] @ nnet['Weights'][i]
        xw_vec, dims = base.to_vector(xw)
        d2 = np.diagflat(base.sigmoid_second_derivative(xw_vec))
        second_derivatives.append(d2)

    output = {'Thetas': thetas,
              'Second_Derivatives': second_derivatives}
    nnet.update(output)
    return nnet



def backward_pass(nnet, vectors):
    n = nnet['Augmented_States'][0].shape[0]
    omega = nnet['Thetas'][-1]
    omegas = [omega]
    Hv_Products = []
    for i in reversed(range(len(nnet['Augmented_Weights']))):
        q = nnet['Weights'][i].shape[1]
        gradient = np.kron(nnet['Augmented_Weights'][i],np.eye(n)) @ nnet['First_Derivatives'][i]
        Hv = np.kron(np.eye(q),nnet['Augmented_States'][i].T) @ nnet['First_Derivatives'][i] \
             @ omega\
             + Dv_Tensor(nnet, vectors, i) \
             + Cv_Tensor(nnet, i)
        new_omega = gradient @ omega + Av_Tensor(nnet, i) + Bv_Tensor(nnet, vectors, i)
        omegas.append(new_omega)
        dims = nnet['Gradients'][i].shape
        Hv = base.to_matrix(Hv, dims)
        Hv_Products.append(Hv)
        omega = new_omega
        #gradfx omega + Bv + A*theta

    omegas.reverse()
    Hv_Products.reverse()

    output = {'Omegas': omegas,
              'Hv_Products': Hv_Products}
    nnet.update(output)

    return nnet

def Av_Tensor(nnet, i):
    vector = nnet['Thetas'][i]
    n = nnet['Augmented_States'][0].shape[0]
    Av = np.kron(nnet['Augmented_Weights'][i],np.eye(n)) \
         @ np.diagflat(nnet['Lambdas'][i+1]) \
         @ nnet['Second_Derivatives'][i] \
         @ np.kron(nnet['Augmented_Weights'][i],np.eye(n)).T \
         @ vector
    return Av

def Bv_Tensor(nnet, vectors, i):
    n = nnet['Augmented_States'][0].shape[0]
    p = nnet['Augmented_Weights'][i].shape[0]
    q = nnet['Augmented_Weights'][i].shape[1]
    Bv = np.kron(nnet['Lambdas'][i+1].T @ nnet['First_Derivatives'][i], np.eye(n*p)) \
         @ K1v_Product(nnet['Weights'][i],n, vectors[i]) \
         + np.kron(nnet['Augmented_Weights'][i], np.eye(n)) \
         @ np.diagflat(nnet['Lambdas'][i+1]) \
         @ nnet['Second_Derivatives'][i] \
         @ np.kron(np.eye(q), nnet['Augmented_States'][i]) \
         @ vectors[i]
    return Bv

"""Need to fix this"""
def Cv_Tensor(nnet, i):
    vector = nnet['Thetas'][i]
    n = nnet['Augmented_States'][0].shape[0]
    p = nnet['Augmented_Weights'][i].shape[0]
    q = nnet['Augmented_Weights'][i].shape[1]
    v = np.kron(nnet['First_Derivatives'][i] @ nnet['Lambdas'][i+1], np.eye(n*p)) @ vector
    Cv = K2v_Product(nnet['Weights'][i], n, v) \
         + (np.kron(nnet['Augmented_Weights'][i], np.eye(n))
         @ np.diagflat(nnet['Lambdas'][i+1])
         @ nnet['Second_Derivatives'][i]
         @ np.kron(np.eye(q), nnet['Augmented_States'][i])).T \
         @ vector
    return Cv

def Dv_Tensor(nnet, vectors, i):
    q = nnet['Augmented_Weights'][i].shape[1]
    Dv = np.kron(np.eye(q), nnet['Augmented_States'][i].T) \
         @ np.diagflat(nnet['Lambdas'][i+1]) \
         @ nnet['Second_Derivatives'][i] \
         @ np.kron(np.eye(q), nnet['Augmented_States'][i]) \
         @ vectors[i]
    return Dv

def K1v_Product(weight, n, vector):
    matrix = base.to_matrix(vector, weight.shape)
    matrix = matrix[1:,]
    vector, dim = base.to_vector(matrix)
    out = np.kron(vector, np.eye(n))
    out, dim = base.to_vector(out)
    return out

def K2v_Product(weight, n, vector):
    p = weight.shape[0] - 1
    q = weight.shape[1]
    vector = base.to_matrix(vector,(n*p*n,q))
    P = np.eye(n*p)
    P, dim = base.to_vector(P)
    P = base.to_matrix(P,(p,n*n*p))
    out = P@vector
    zero = np.zeros((1,q))
    out = np.vstack((zero,out))
    out, dims = base.to_vector(out)
    return out
