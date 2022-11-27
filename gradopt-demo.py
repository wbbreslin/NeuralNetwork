import numpy as np
from gradopt import *

def f(x, a):
    """ Rosenbrock test function with domain R^2 parameter a:
        f(x) = a (x2 - x1^2)^2 + (1 - x1)^2
    """
    # return f(x)
    return 100.*(x[1]-x[0]*x[0])**2 + (1-x[0])**2

def df(x, a):
    """ Returns the gradient of the Rosenbrock function
    """
    # compute partial derivatives
    df_dx0 = (4*a) * x[0]**3 - (4*a) * x[0] * x[1] + 2*x[0] - 2
    df_dx1 = -(2*a) * x[0]**2 + (2*a) * x[1]
    # return df(x)
    return np.array([df_dx0, df_dx1])

def d2f(x, a):
    """ Returns the Hessian of the Rosenbrock function
    """
    # compute partial derivatives
    d2f_d2x1 = (12*a) * x[0]**2 - (4*a) * x[1] + 2
    d2f_d2x1x2 = -(4*a) * x[0]
    d2f_d2x2x1 = d2f_d2x1x2
    d2f_d2x2 = 2*a
    # return d2f(x)
    return np.array([[d2f_d2x1, d2f_d2x1x2],[d2f_d2x2x1, d2f_d2x2]])

# initialize x0
# for neural net, initialize x0 = (W0, b0) as a column
x0 = np.zeros(2)

# set stepdir arg tuples
# for neural net, use f, df, and d2f for that cost function
# they are called like f(x, f_args) where x = (W, b) (a column)
f_args = (100.,)
df_args = (100.,)
d2f_args = (100.,)
steepest_args = (df, df_args)
newton_args = (f, f_args, df, df_args, d2f, d2f_args)

# set backtracking stepsize arg tuple
# for neural net, no changes
alpha = 1.; rho = 0.5; c = 0.01 # for testing Armijo rule (Wolfe 1)
backtrack_args = (f, f_args, df, df_args, alpha, rho, c)

# perform steepest descent optimization using backtracking line search
# for neural net, no changes
s_iters, s_f, s_df = graditer(x0, f, f_args, df, df_args, \
                              steepest_dir, steepest_args, \
                              backtrack_stepsize, backtrack_args)
print('steepest descent iterations: %d' % s_iters.size)
print('x^* =', s_iters[-1])
print('f(x^*) =', s_f[-1])
print('|df(x^*)| = ', s_df[-1], '\n')

# perform Newton's method optimization using backtracking line search
# for neural net, no changes
n_iters, n_f, n_df = graditer(x0, f, f_args, df, df_args, \
                              newton_dir, newton_args, \
                              backtrack_stepsize, backtrack_args)
print('Newton\'s method iterations: %d' % n_iters.size)
print('x^* =', n_iters[-1])
print('f(x^*) =', n_f[-1])
print('|df(x^*)| = ', n_df[-1], '\n')
