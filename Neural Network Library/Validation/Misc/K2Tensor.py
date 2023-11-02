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
    print((S@A).shape)
    print(T.shape)
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

#Do not go bigger...
n = 2

weight = np.array([[1,2,3],
                   [5,6,7],
                   [5,6,7]])

vec, dim = base.to_vector(weight)


#RUNTIME: 24.790450749977026 seconds
start = timeit.default_timer()
direct = Kron_Tensors(weight, n)
print(direct.shape)
#direct = direct @ vec
stop = timeit.default_timer()
direct_time = stop-start

start = timeit.default_timer()
indirect = som.Kv_Product(weight,n,vec)
stop = timeit.default_timer()
indirect_time = stop-start

'''
print("Error between results:")
print(np.sum(indirect-direct))
print("Indirect Method Runtime:")
print(indirect_time)
print("Direct Method Runtime:")
print(direct_time)
'''

#n = 2, p = 2, q = 3
p = weight.shape[0]-1
q = weight.shape[1]
vector = np.array([i for i in range(1, n*n*q*p+ 1)]).reshape(-1,1)
print(base.to_matrix(vector,(n*p*n,q)))
#print(M.shape)
print(direct.T@vector)
#W = np.array([[1,4,7],[2,5,8],[3,6,9]])
#P = np.eye(3)[:, ::-1]

#This works for my current example, need to generalize
P = np.eye(4)[:, [0,3,2,1]]
P, dim = base.to_vector(P)
P = base.to_matrix(P, ((n,n*n*p)))
print(P)