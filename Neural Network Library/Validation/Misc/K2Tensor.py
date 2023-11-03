import SecondOrderModel as som
import timeit
import numpy as np
import Base as base
from scipy.sparse import coo_matrix, kron

"""
Old Method: Kron_Tensors() and generate_matrix()
New Method: K2v_Product()

Objective:
1) Compute K.T @ v
2) Measure the error to ensure accuracy of new method to the exact old method
3) Compare runtimes between methods

Results:
1) New method generates exact same output as old method
2) Runtime of new method for n=1000, p=2, q=4 is  0.01698 seconds
3) Runtime of old method for n=1000, p=2, q=4 is 32.72858 seconds
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

def K2v_Product(weight, n, vector):
    p = weight.shape[0] - 1
    q = weight.shape[1]
    vector = base.to_matrix(vector,(n*p*n,q))
    P = np.eye(n*p)
    P, dim = base.to_vector(P)
    P = base.to_matrix(P,((p,n*n*p)))
    out = P@vector
    zero = np.zeros((1,q))
    out = np.vstack((zero,out))
    out, dims = base.to_vector(out)
    return(out)

n = 1000 #Do not go bigger runtime for old method is p*n**3

weight = np.array([[1,2,3,4],
                   [5,6,7,4],
                   [5,6,7,4]])

vec, dim = base.to_vector(weight)


#RUNTIME: 24.790450749977026 seconds for n=1000
start = timeit.default_timer()
direct = Kron_Tensors(weight, n)
#direct = direct @ vec
stop = timeit.default_timer()
direct_time = stop-start


#n = 2, p = 2, q = 3
p = weight.shape[0]-1
q = weight.shape[1]

'''Create a dummy vector'''
vector = np.random.rand(n*n*p*q,1)

"""K.Tv Product From direct method"""
DirectMethod = direct.T @ vector


start = timeit.default_timer()
FastMethod = K2v_Product(weight, n, vector)
stop = timeit.default_timer()
indirect_time = stop-start
error = np.sum((DirectMethod-FastMethod)**2)

print("Error between results:")
print(error)
print("Indirect Method Runtime:")
print(indirect_time)
print("Direct Method Runtime:")
print(direct_time)
