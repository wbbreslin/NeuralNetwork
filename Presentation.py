"""Examples Convergence vs non-convergence"""

W,b,cost = descent_NN_Model(x,y,
                            neurons=[2,2,3,2],
                            eta=0.05,
                            itr=10**4,
                            stop=False)

W,b,cost = descent_NN_Model(x,y,
                            neurons=[2,2,3,2],
                            eta=0.05,
                            itr=3*10**4,
                            stop=False)

W,b,cost = stochastic_NN_Model(x,y,
                               neurons=[2,2,3,2],
                               eta = 0.05,
                               itr = 3*10**5,
                               stop=False)

"""Compare performance"""
W,b,cost = backtracking_descent_NN_Model(x,y,
                                         neurons=[2,2,3,2],
                                         itr=10**6,
                                         stop=True)


W,b,cost = descent_NN_Model(x,y,
                            neurons=[2,2,3,2],
                            eta=0.05,
                            itr=10**6,
                            stop=True)



W,b,cost = stochastic_NN_Model(x,y,
                               neurons=[2,2,3,2],
                               eta = 0.05,
                               itr = 10**6,
                               stop=True)

"""Compare backtrack to descent with bigger step size"""

W,b,cost = backtracking_descent_NN_Model(x,y,
                                         neurons=[2,2,3,2],
                                         itr=10**6,
                                         stop=True)

W,b,cost = descent_NN_Model(x,y,
                            neurons=[2,2,3,2],
                            eta=0.5,
                            itr=10**6,
                            stop=True)
