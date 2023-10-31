import numpy as np
from scipy.sparse import coo_matrix, kron
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



def backward_pass(nnet, vectors, KTensors):
    n = nnet['Augmented_States'][0].shape[0]
    omega = nnet['Thetas'][-1]
    omegas = [omega]
    for i in reversed(range(len(nnet['Augmented_Weights']))):
        gradient = np.kron(nnet['Augmented_Weights'][i],np.eye(n)) @ nnet['First_Derivatives'][i]
        new_omega = gradient @ omega + Av_Tensor(nnet, i) + Bv_Tensor(nnet, vectors, KTensors, i)
        omegas.append(new_omega)
        omega = new_omega
        #gradfx omega + Bv + A*theta

    omegas.reverse()
    output = {'Omegas': omegas}
    nnet.update(output)

    return nnet

def Av_Tensor(nnet, i):
    vector = nnet['Thetas'][i]
    n = nnet['Augmented_States'][0].shape[0]
    Av = np.kron(nnet['Augmented_Weights'][i],np.eye(n)) \
         @ np.diagflat(nnet['Lambdas'][i+1]) \
         @ nnet['Second_Derivatives'][i] \
         @ np.kron(nnet['Augmented_Weights'][i].T,np.eye(n)) \
         @ vector
    return Av

def Bv_Tensor(nnet, vectors, KTensors, i):
    n = nnet['Augmented_States'][0].shape[0]
    p = nnet['Augmented_Weights'][i].shape[0]
    q = nnet['Augmented_Weights'][i].shape[1]
    Bv = np.kron(nnet['Lambdas'][i+1].T @ nnet['First_Derivatives'][i], np.eye(n*p)) \
         @ KTensors[i] \
         @ vectors[i] \
         + np.kron(nnet['Augmented_Weights'][i], np.eye(n)) \
         @ np.diagflat(nnet['Lambdas'][i+1]) \
         @ nnet['Second_Derivatives'][i] \
         @ np.kron(np.eye(q), nnet['Augmented_States'][i]) \
         @ vectors[i]
    return Bv

def Dv_Tensor_Old(vector, state, Lambda, second_derivative, q):
    # may need to vec the Lambdas before diagflat
    Dv = np.kron(np.eye(q), state.T) \
         @ np.diagflat(Lambda) \
         @ second_derivative \
         @ np.kron(np.eye(q),state) \
         @ vector
    return Dv


def Kron_Tensors(weight, n):
    dimensions = weight.shape
    p = dimensions[0]-1
    q = dimensions[1]

    A1 = np.zeros((p, 1))
    A2 = np.eye(p)
    A = np.hstack((A1, A2))

    Kv = coo_matrix((n*q*n*p,(p+1)*q))
    print(Kv.shape)
    for i in range(n):
        S = generate_matrix(n,p,i)
        T = generate_matrix(n,q,i)
        Kv = Kv + kron(T,S @ A)
    return Kv

def generate_matrix(n, dim, itr):
    rows = n*dim
    columns = dim
    data, row_indices, col_indices = [], [], []
    for i in range(rows):
        if i % n == itr:
            row_indices.append(i)
            col_indices.append(i // n)
            data.append(1)

    S = coo_matrix((data, (row_indices, col_indices)), shape=(rows, columns))
    return S

"""
weight = np.array([[1,2,3,4],
              [5,6,7,8],
              [9,10,11,12]])

S1 = generate_matrix(3,3,0)
S2 = generate_matrix(2,3,1)
T1 = generate_matrix(2,4,0)
T2 = generate_matrix(2,4,1)

"""
