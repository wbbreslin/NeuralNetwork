import numpy as np
#import matplotlib
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['font.size']=14

from matplotlib import pyplot as plt
from scipy.special import expit

from gradopt import *

def model(x, W, b):
    """ returns classification for the data point x """
    _x = x
    for l in range(1,len(W)):
        _x = expit(W[l]@_x + b[l])
    return _x

def perf(data_x, train_y, W, b):
    """ for a W, b pair, reports on classification performance """
    correct = 0
    N = data_x.shape[1]
    for n in range(N):
        model_x = np.round(model(data_x[:, n], W, b))
        err_str = " "
        if (model_x == train_y[:, n]).all():
            correct += 1
        else:
            err_str = "x"
        print(err_str, model_x, ", ", train_y[:, n])
    print('\nperformance: %.1f%% (%d / %d)' % (100 * correct / N, correct, N))

def W_b_vec_to_list(W_b, dims):
    """ converts a W, b column vector to W, b lists of arrays """
    layers = len(dims)
    _W = [0]
    i0 = 0
    # unpack Ws
    for l in range(layers-1):
        i1 = i0 + dims[l+1] * dims[l]
        #print('i0:', i0, ' i1:',i1)
        _W += [ W_b[i0:i1].reshape((dims[l+1], dims[l]), order='F') ]
        i0 = i1
    # unpack bs
    _b = [0]
    for l in range(1, layers):
        i1 = i0 + dims[l]
        #print('i0:', i0, ' i1:', i1)
        _b += [ W_b[i0:i1] ]
        i0 = i1
    return _W, _b

def W_b_list_to_vec(W, b):
    """ converts W, b lists of arrays to a W, b column vector """
    nd = 0
    for l in range(1, len(W)):
        nd += W[l].size + b[l].size
    _x = np.zeros(nd)
    i = 0
    for l in range(1, len(W)):
        _W = W[l].flatten('F')
        _x[i:i+_W.size] = _W
        i += _W.size
    for l in range(1, len(b)):
        _b = b[l]
        _x[i:i+_b.size] = _b
        i += _b.size
    return _x

def F(x, dims, data_x, train_y, W_b_vec_to_list, model):
    """ for a W, b column vector x, returns F(x) for the neural net problem """
    W, b = W_b_vec_to_list(x, dims)
    N = data_x.shape[1]
    cost = 0
    for n in range(N):
        cost += 0.5*np.linalg.norm(train_y[:, n] - model(data_x[:, n], W, b))**2
    return cost / N

def dF(x, dims, data_x, train_y, W_b_vec_to_list, W_b_list_to_vec):
    """ for a W, b column vector x, returns dF(x) for the neural net problem """
    _W, _b = W_b_vec_to_list(x, dims)
    dF_dW = [0] #; _dF_dW = [0]
    dF_db = [0] #; _dF_db = [0]
    layers = len(dims)
    N = data_x.shape[1]
    for n in range(N):
        z = [0.]
        a = [data_x[:,n]]
        y = train_y[:,n] #; y=_y[:,d]
        # forward propagation
        for l in range(1, layers):
            z.append(_W[l]@a[l-1]+_b[l])
            a.append(expit(z[l]))
        sigma_z = expit(z[-1])
        # set delta^L
        delta = [sigma_z*(1-sigma_z)*(a[-1]-y)]
        # backward propagation, set delta^l
        for l in range(layers-2, 0, -1):
            sigma_z = expit(z[l])
            # prepend delta^l using previous (next) delta^{l+1} value
            delta = [sigma_z*(1-sigma_z)*(_W[l+1].T@delta[0])] + delta
        delta = [0.] + delta
        # build grad C for current datum
        _dF_dW = [0]
        _dF_db = [0]
        for l in range(1, layers):
            # initialize current datum grad_W C for current layer
            _dF_dW = _dF_dW + [np.zeros((dims[l],dims[l-1]))]
            # fill in components of grad_W C for current datum, current layer
            for j in range(dims[l]):
                for k in range(dims[l-1]):
                    _dF_dW[l][j,k] = delta[l][j]*a[l-1][k]
                next
            # set current datum grad_b C for current layer
            _dF_db = _dF_db + [delta[l]]
            # if we're on the first datum, grad C is grad_W C, grad_b C
            if n == 0:
                dF_dW.append(_dF_dW[l])
                dF_db.append(_dF_db[l])
            # else, add grad_W C, grad_b C for current datum to existing
            # derivative by linearity
            else:
                dF_dW[l] += _dF_dW[l]
                dF_db[l] += _dF_db[l]
    return W_b_list_to_vec(dF_dW, dF_db)

# initialize input data
x1 = np.array([0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7])
x2 = np.array([0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6])
data_x = np.zeros((2, x1.size))
data_x[0,:] = x1; data_x[1,:] = x2

# initialize training data
train_y = np.zeros((2,10)); train_y[0,:5] = 1.; train_y[1,5:] = 1.
rnd = np.random.default_rng(seed=1123).standard_normal

# set number of layers and dimensions of each layer
# dims[0] == input layer, dims[-1] == output layer
dims = (2, 2, 3, 2)
layers = len(dims)

# initialize Ws and bs (lists of weights and biases, 0 for input layer)
W = [0]; b = [0]
for l in range(1, layers):
    W.append(0.5*rnd((dims[l], dims[l-1])))
    b.append(0.5*rnd(dims[l]))

# initialize x0 as Ws and bs as a column (W1[:], ..., Wl[:], b1[:], ... bl[:])
x0 = W_b_list_to_vec(W, b)

# arguments required for F and dF
F_args = (dims, data_x, train_y, W_b_vec_to_list, model)
dF_args = (dims, data_x, train_y, W_b_vec_to_list, W_b_list_to_vec)

# arguments for direction functions
steepest_args = (dF, dF_args)

# arguments for constant step size function
stepsize = 0.5
const_args = (stepsize, )

# arguments for backtrack step size function
alpha = 1.; rho = 0.5; c = 0.01
backtrack_args = (F, F_args, dF, dF_args, alpha, rho, c) # backtrack step size

# perform gradient descent using constant step size
c_x_iters, c_f, c_df = graditer(x0, F, F_args, dF, dF_args, \
                                steepest_dir, steepest_args, \
                                const_stepsize, const_args, \
                                eps=1.0e-3)
print('\nconstant step size iterations: %d' % len(c_x_iters))
print('f(x^*) = %f' % c_f[-1])
print('|df(x^*)| = %f' % c_df[-1])
W, b = W_b_vec_to_list(c_x_iters[-1], dims)
perf(data_x, train_y, W, b)

# perform gradient descent using backtrack step size
b_x_iters, b_f, b_df = graditer(x0, F, F_args, dF, dF_args, \
                                steepest_dir, steepest_args, \
                                backtrack_stepsize, backtrack_args, \
                                eps=1.0e-3)
print('\nbacktrack iterations: %d' % len(b_x_iters))
print('f(x^*) = %f' % b_f[-1])
print('|df(x^*)| = %f' % b_df[-1])
W, b = W_b_vec_to_list(b_x_iters[-1], dims)
perf(data_x, train_y, W, b)
