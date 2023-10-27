import numpy as np
from scipy.sparse import coo_matrix, kron

def forward_second_pass(x):
    return x

def backward_second_pass(x):
    return x
def Av_Tensor(vector, weight, Lambda, second_derivative, n):
    #may need to vec the Lambdas before diagflat
    Av = np.kron(weight,np.eye(n)) \
         @ np.diagflat(Lambda) \
         @ second_derivative \
         @ np.kron(weight.T,np.eye(n)) \
         @ vector
    return Av

def Bv_Tensor(vector, state, weight, Lambda, K, first_derivative, second_derivative):
    n = state.shape[0]
    p = weight.shape[0]
    q = weight.shape[1]
    Bv = np.kron(first_derivative.T @ np.diagflat(Lambda), np.eye(n*p)) \
         @ K \
         @ vector \
         + np.kron(weight, np.eye(n)) \
         @ np.diagflat(Lambda) \
         @ second_derivative \
         @ np.kron(np.eye(q), state) \
         @ vector
    return Bv
def Dv_Tensor(vector, state, Lambda, second_derivative, q):
    # may need to vec the Lambdas before diagflat
    Dv = np.kron(np.eye(q), state.T) \
         @ np.diagflat(Lambda) \
         @ second_derivative \
         @ np.kron(np.eye(q),state) \
         @ vector
    return Dv


def Kv_Tensor(weight, n):
    p = weight.shape[0]
    q = weight.shape[1]
    Kv = coo_matrix((n*q*n*p,p*q))
    for i in range(n):
        S = generate_matrix(n,p,i)
        T = generate_matrix(n,q,i)
        Kv = Kv + kron(T,S)
    return Kv

def generate_matrix(n, dim, itr):
    rows = n*dim
    columns = dim
    data, row_indices, col_indices = [], [], []
    for i in range(rows):
        if i % n == itr:
            row_indices.append(i)
            col_indices.append(i // 2)
            data.append(1)

    S = coo_matrix((data, (row_indices, col_indices)), shape=(rows, columns))
    return S

"""
weight = np.array([[1,2,3,4],
              [5,6,7,8],
              [9,10,11,12]])

S1 = generate_matrix(2,3,0)
S2 = generate_matrix(2,3,1)
T1 = generate_matrix(2,4,0)
T2 = generate_matrix(2,4,1)
"""