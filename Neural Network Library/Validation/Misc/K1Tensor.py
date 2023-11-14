import SecondOrderModel as som
import timeit
import numpy as np
import Base as base
from scipy.sparse import coo_matrix, kron

"""
Old Method
"""

def Kron_Tensors(weight, n):
    dimensions = weight.shape
    p = dimensions[0]-1
    q = dimensions[1]

    A1 = np.zeros((p, 1))
    A2 = np.eye(p)
    A = np.hstack((A1, A2))

    Kv = coo_matrix((n*q*n*p,(p+1)*q))
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


n = 1000 #Do not go bigger...

weight = np.array([[1,4,7,10],
                   [2,5,8,11],
                   [3,6,9,12]])

vec = base.to_vector(weight)

print(vec)
#RUNTIME: 24.790450749977026 seconds
start = timeit.default_timer()
direct = Kron_Tensors(weight, n)
print(direct.toarray())
direct = direct @ vec
print(direct)
stop = timeit.default_timer()
direct_time = stop-start

start = timeit.default_timer()
indirect = som.K1v_Product(weight,n,vec)
stop = timeit.default_timer()
indirect_time = stop-start
error = np.sum((indirect-direct)**2)

print("Error between results:")
print(error)
print("Indirect Method Runtime:")
print(indirect_time)
print("Direct Method Runtime:")
print(direct_time)