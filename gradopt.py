#!/usr/bin/env python

import numpy as np

### stepsize functions
### arguments for all stepsize functions are:
### x_k, f_k, p_k - required
### additional required arguments - optional
### keyword argument, containing 'df_k' == df(x_k) if it is already known

def const_stepsize(x_k, f_k, p_k, alpha, f_evals, df_evals, d2f_evals, **kwargs):
    """ Constant step size function

        Inputs
        x_k   : Current iterate; for compatibility - unused
        f_k   : Current value f_k; for compatibility - unused
        p_k   : Search direction; for compatibility - unused
        alpha : Constant step size value (float)

        kwargs : Optional keyword args:
                 For compatibility - unused

        Outputs
        alpha  : Constant step size
        x_k    : Current iterate since we don't know what the next will be
        f_k    : Cost function value at current iterate
    """
    # return constant step size, current iterate x_k, current f val f_k
    return alpha, x_k, f_k


def backtrack_stepsize(x_k, f_k, p_k, f, f_args, df, df_args, alpha, rho, c, f_evals, df_evals, d2f_evals, **kwargs):
    """ Backtracking step size function - for a given iterate x_k
        and direction p_k, it finds and returns a step size such
        that the first Wolfe condition W1 is satisfied. It also
        returns the next iterate and the value of the cost function
        at the next iterate since they are both computed while
        searching for the next step size.

        Inputs
        x_k     : Current iterate
        f_k     : Cost function value at current iterate
        p_k     : Search direction
        f       : Cost function, returns f(x) (function)
        f_args  : Tuple of args for the cost function; cost function is
                  called by f(x, *f_args)
        df      : Gradient of cost function, returns df(x) (function)
        df_args : Tuple of args for the gradient function; gradient function
                  is called by df(x, *df_args)
        alpha   : Initial step size
        rho     : Backtracking factor for scaling alpha if the
                  Wolfe condition W1 is not satisfied
        c       : Constant for the first Wolfe condition W1

        kwargs : Optional keyword args:
                 df_k  : Holds df_k if it is already known
   
        Outputs:
        alpha        : Step size from x_k in direction p_k such that
                       the first Wolfe condition W1 is satisfied
        x_k1         : Next iterate
        f_k1         : Cost function value at next iterate
    """
    # obtain df(x_k) for checking first Wolfe condition W1
    if 'df_k' in kwargs.keys():
        df_k = kwargs['df_k']
    else:
        df_k = df(x_k, *df_args)
        df_evals[-1] += 1

    # scale step size by rho until first Wolfe condition W1 is satisfied
    x_k1 = x_k + alpha * p_k
    f_k1 = f(x_k1, *f_args)
    f_evals[-1] += 1
    c_p_df = c * p_k @ df_k
    while f_k1 > (f_k + alpha * c_p_df):
        alpha = rho * alpha
        x_k1 = x_k + alpha * p_k
        f_k1 = f(x_k1, *f_args)
        f_evals[-1] += 1

    # return step size, next iterate x_{k+1}, f(x_{k+1})
    return alpha, x_k1, f_k1


### next direction functions
### arguments for all step direction functions are:
### x_k - required
### additional required arguments - optional
### keyword argument, containing 'df_k' == df(x_k) if it is already known

def steepest_dir(x_k, df, df_args, f_evals, df_evals, d2f_evals,**kwargs):
    """ Function for calculating the steepest descent search direction

        Inputs
        x_k     : Point at which to calculate the gradient
        df      : Gradient of cost function, returns df(x) (function)
        df_args : Tuple of args for the gradient function; gradient function
                  is called by df(x, *df_args)


        kwargs : Optional keyword args:
                 df_k  : Holds df_k if it is already known

        Outputs
        Steepest descent search direction -df(x_k)
    """
    # return steepest descent direction at x_k
    if 'df_k' in kwargs.keys():
        return -kwargs['df_k']
    df_evals[-1] += 1
    return -df(x_k, *df_args)


def newton_dir(x_k, f, f_args, df, df_args, d2f, d2f_args, f_evals, df_evals, d2f_evals, **kwargs):
    """ Function for calculating the Newton's method search direction

        Inputs
        f        : Cost function, returns f(x) (function)
        f_args   : Tuple of args for the cost function; cost function is
                   called by f(x, *f_args)
        df       : Gradient of cost function, returns df(x) (function)
        df_args  : Tuple of args for the gradient function; gradient function
                   is called by df(x, *df_args)
        d2f      : Hessian of cost function, returns d2f(x) (function)
        d2f_args : Tuple of args for the Hessian function; Hessian function
                   is called by d2f(x, *d2f_args)
        x_k : Point at which to calculate the gradient and Hessian

        kwargs : Optional keyword args:
                 df_k  : Holds df_k if it is already known

        Outputs
        Newton's method search direction -(d2f(x_k))^{-1}(df(x_k)) as the
        solution to the linear system -d2f(x_k)x=df(x_k)
    """
    # return Newton's method direction at x_k
    if 'df_k' in kwargs.keys():
        df_k = kwargs['df_k']
    else:
        df_k = df(x_k, *f_args)
        df_evals[-1] += 1
    hess = d2f(x_k, *d2f_args)
    d2f_evals[-1] += 1
    # use lstsq instead of solve in case Hessian is singular
    x, resid, ran, s = np.linalg.lstsq(hess, df_k, rcond=None)
    return -x
    #return -np.linalg.solve(d2f(x_k, *d2f_args), df_k)


### general gradient descent algorithm

def graditer(x_k, f, f_args, df, df_args, stepdir, stepdir_args, stepsize, stepsize_args, \
        eps=1.0e-5, maxiter=1.0e+6, F2=None):
    """ graditer(): Iterative line search optimization algorithm

        This solves an optimization problem where at each point one
        chooses a direction and a step size. It can use any function
        for choosing a step direction and any function for choosing a
        step size at a given point x_k.

        Inputs
        x_k           : Starting point
        f             : Cost function, returns f(x) (function)
        f_args        : Tuple of args for the cost function; cost function is
                        called by f(x, *f_args)
        df            : Gradient of cost function, returns df(x) (function)
        df_args       : Tuple of args for the gradient function; gradient
                        function is called by df(x, *df_args)
        stepdir       : Function for computing a step direction at the current
                        iterate x_k (function)
        stepdir_args  : Arguments required for step direction function (tuple)
        stepsize      : Function for returning step size alpha at the current
                        iterate x_k and direction p_k (function)
        stepsize_args : Arguments required for step size function (tuple)
        eps           : Stopping criterion - stop if norm of gradient is
                        less than eps (optional - default 1.0e-5) (float)
        maxiter       : Stopping criterion - stop if number of iterations
                        reaches maxiter (optional - default 1.0e+6) (float)

        Outputs
        iters         : 2d array of iterates {x_k} (x^* = iters[-1])
        f_vals        : Array of f values {f(x_k)} (f(x^*) = f_vals[-1])
        norm_df_vals  : Array of |df| values {|df(x_k)|}
    """
    # initialize niter (# iterations) and iters list (stores iterate data)
    # and store initial iterate, f_k, and |df_k| vals
    niter = 0
    iters = np.array([x_k])
    f_k = f(x_k, *f_args)
    f_evals = np.array([1])
    f_vals = np.array([f_k])
    df_k = df(x_k, *df_args)
    df_evals = np.array([1])
    d2f_evals = np.array([0])
    norm_df_k = np.linalg.norm(df_k)
    norm_df_vals = np.array([norm_df_k])

    # begin loop
    while niter < maxiter:
        f_evals = np.append(f_evals, 0)
        df_evals = np.append(df_evals, 0)
        d2f_evals = np.append(d2f_evals, 0)
        #stepdir_args['x_k'] = x_k
        p_k = stepdir(x_k, *stepdir_args, \
                f_evals, df_evals, d2f_evals, \
                df_k=df_k, niter=niter)
        # get next step size
        alpha, _x_k1, _f_k1 = stepsize(x_k, f_k, p_k, *stepsize_args, \
                f_evals, df_evals, d2f_evals, \
                df_k=df_k)
        # calculate next iterate
        if (_x_k1 != x_k).all():
            x_k = _x_k1
        else:
            x_k = x_k + alpha * p_k
        # update values for next loop and store current state
        if _f_k1 != f_k:
            f_k = _f_k1
        else:
            f_k = f(x_k, *f_args)
            f_evals[-1] += 1
        df_k = df(x_k, *df_args)
        df_evals[-1] += 1
        norm_df_k = np.linalg.norm(df_k)
        iters = np.append(iters, [x_k], axis=0)
        f_vals = np.append(f_vals, f_k)
        norm_df_vals = np.append(norm_df_vals, norm_df_k)
        # increment number of iterations
        niter += 1
        # stop if |df(x_k)| < eps
        if norm_df_k < eps or (F2 and F2(x_k, *f_args) == 0):
            break
    # return results
    return iters, f_vals, norm_df_vals, f_evals, df_evals, d2f_evals
