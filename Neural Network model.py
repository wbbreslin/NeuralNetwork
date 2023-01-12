import numpy as np
import matplotlib.pyplot as plt
from itertools import product

"""Creating the data set"""
x1 = np.array([0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7])
x2 = np.array([0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6])
y1 = np.hstack([np.ones(5), np.zeros(5)])
y2 = np.hstack([np.zeros(5), np.ones(5)])
y = np.vstack([y1,y2])
x = np.transpose(np.vstack([x1,x2]))
y = np.transpose(y)

"""
Things to do:
* Add inverse function to go from [0,1] to R^n
* Rename stochastic gradient function, as it is used in non stochastic NN models
* Work on incorporating the hessian for Newton method
* Modify code to work for response surface DoE models
"""

def activate(x,W,b):
    """Sigmoid activation function"""
    z = W@x+b
    a = 1/(1+np.exp(-z))
    return(a)

def backtrackingLineSearch(x,y,W,b,dW,db,alpha,rho,c):
    """Backtracking search method for gradient descent"""
    # Concatenate Lists
    theta0 = W+b
    gradient = dW+db
    # Convert to Vectors
    theta0_vec, dim_theta = toVector(theta0)
    gradient_vec, dim_gradient = toVector(gradient)
    # Define terms, calculate cost functions
    cost_old = costFunction(x,y,W,b)
    theta1_vec = theta0_vec - alpha*gradient_vec
    theta1 = toMatrix(theta1_vec,dim_theta)
    W1 = theta1[0:len(W)]
    b1 = theta1[len(W):]
    cost_new = costFunction(x,y,W1,b1)
    cp_df = -c*gradient_vec@gradient_vec
    # Line Search Algorithm
    while cost_new > (cost_old + alpha*cp_df):
        alpha = rho * alpha
        theta1_vec = theta0_vec - alpha*gradient_vec
        theta1 = toMatrix(theta1_vec,dim_theta)
        W1 = theta1[0:len(W)]
        b1 = theta1[len(W):]
        cost_new = costFunction(x,y,W1,b1)
    return(alpha)

def backtracking_descent_NN_Model(x,y,neurons,itr,stop):
    """Neural Network model using backtracking gradient descent method"""
    W,b = createNetwork(neurons)
    N = x.shape[0]
    cost = np.zeros(itr)
    result = False
    for i in range(itr):
        dW = resetParameters(W)
        db = resetParameters(b)
        for k in range(N):
            x0, y0 = selectXY(x,y,k)
            a = forwardPass(x0,W,b)
            delta = backwardPass(y0,a,W)
            dW0, db0 = stochasticGradient(x0,a,delta)
            for j in range(len(W)):
                dW[j] += dW0[j]
                db[j] += db0[j]
        eta = backtrackingLineSearch(x,y,W,b,dW,db,alpha=1,rho=0.5,c=0.01)
        W = updateParameters(W,dW,eta)
        b = updateParameters(b,db,eta)
        newcost = costFunction(x,y,W,b)
        cost[i] = newcost
        result = checkResult(x,y,W,b)
        if stop == True:
            if result == True:
                print(i)
                break;
    return(W,b,cost)

def backwardPass(y,a,W):
    """Backward pass through the network to calculate deltas"""
    layers = len(W)
    delta = []
    gradient = []
    for i in reversed(range(layers)):
        j = layers-i-1
        D = np.diagflat(a[i]*(1-a[i]))
        if i == layers-1:
            delta.append(D@(a[i]-y))
        else:
            delta.append(D@(np.transpose(W[i+1])@delta[j-1]))
    delta.reverse()
    return(delta)

def checkResult(x,y,W,b):
    """The cost function to be minimized"""
    n = x.shape[0]
    costvec = np.zeros(n)
    a_vec=[]
    for i in range(n):
        data = selectXY(x,y,i)
        x0 = data[0]
        y0 = data[1]
        a = forwardPass(x0,W,b)
        a_vec.append(a[-1])
        costvec[i] = 0.5*np.linalg.norm(y0-np.around(a[-1]))**2
    cost = sum(costvec)/10
    if cost == 0:
        return(True)
    else:
        return(False)

def costFunction(x,y,W,b):
    """The cost function to be minimized"""
    n = x.shape[0]
    costvec = np.zeros(n)
    for i in range(n):
        data = selectXY(x,y,i)
        x0 = data[0]
        y0 = data[1]
        a = forwardPass(x0,W,b)
        costvec[i] = 0.5*np.linalg.norm(y0-a[-1])**2
    cost = sum(costvec)/10
    return(cost)
    
def createNetwork(neurons):
    """Initiates Network parameters given a list of neurons per layer"""
    W = []; b = []
    np.random.seed(100)
    for i in range(len(neurons)-1):
        W.append(0.5*np.random.rand(neurons[i+1],neurons[i]))
        b.append(0.5*np.random.rand(neurons[i+1],1))
    return(W,b)

def descent_NN_Model(x,y,neurons,eta,itr,stop):
    """Neural Network model using gradient descent method"""
    W,b = createNetwork(neurons)
    N = x.shape[0]
    cost = np.zeros(itr)
    for i in range(itr):
        dW = resetParameters(W)
        db = resetParameters(b)
        for k in range(N):
            x0, y0 = selectXY(x,y,k)
            a = forwardPass(x0,W,b)
            delta = backwardPass(y0,a,W)
            dW0, db0 = stochasticGradient(x0,a,delta)
            for j in range(len(W)):
                dW[j] += dW0[j]
                db[j] += db0[j]
        gradient = [dW,db]
        W = updateParameters(W,dW,eta)
        b = updateParameters(b,db,eta)
        newcost = costFunction(x,y,W,b)
        cost[i] = newcost
        result = checkResult(x,y,W,b)
        if stop==True:
            if result == True:
                print(i)
                break;
    return(W,b,cost)

def forwardPass(x,W,b):
    """Passes a single data point through the network"""
    a = []
    for i in range(len(W)):
        if i == 0:
            a.append(activate(x,W[i],b[i]))
        else:
            a.append(activate(a[i-1],W[i],b[i]))
    return(a)

def plotCost():
    """Plot the log-cost against the iterates"""
    plt.plot(np.array(range(len(cost))),np.log10(cost))
    plt.title("Convergence of Cost Function")
    plt.xlabel("Iteration")
    plt.ylabel("Log Cost")
    plt.show()

def plotRegion():
    """Plot of the predicted region partition"""
    xh = np.arange(0,1,0.01/4)
    xv = np.arange(0,1,0.01/4)
    xall = np.array(list(product(xh, xv)))
    ySave = xall*0
    for i in range(xall.shape[0]):
        xp = np.array([[xall[i,0]],
                      [xall[i,1]]])
        a = forwardPass(xp,W,b)
        ySave[i,0]=a[-1][0]
        ySave[i,1]=a[-1][1]
    ySave = np.around(ySave,0)
    yc = ySave[:,0]
    plt.scatter(xall[:,0]*yc, xall[:,1]*yc, s=1, color='gray')
    plt.scatter(x1[:5], x2[:5], marker='o', color='b')
    plt.scatter(x1[5:], x2[5:], marker='o', color='r')
    plt.show()

def resetParameters(par):
    """Reset list of parameters to zeros (for gradient calculation)"""
    update = []
    for i in range(len(par)):
        dim = par[i].shape
        update.append(np.zeros(dim))
    return(update)

def selectXY(x,y,k):
    """Selects an (x,y) pair based on index k as an np.array"""
    n = x.shape[0]
    x0 = np.array([[x[k,0]],
                   [x[k,1]]])
    y0 = np.array([[y[k,0]],
                   [y[k,1]]])
    return(x0,y0)

def stochasticGradient(x,a,delta):
    """Gradient calculation for stochastic model"""
    layers = len(delta)
    wGradient = []
    bGradient = delta
    for i in range(layers):
        if i==0:
            wGradient.append(delta[i]@np.transpose(x))
        else:
            wGradient.append(delta[i]@np.transpose(a[i-1]))
    return(wGradient,bGradient)

def stochastic_NN_Model(x,y,neurons,eta,itr,stop):
    """Neural Network model using the Stochastic Gradient method"""
    W,b = createNetwork(neurons)
    N = x.shape[0]
    cost = np.zeros(itr)
    result = False
    for i in range(itr):
        k = np.random.randint(N)
        x0, y0 = selectXY(x,y,k)
        a = forwardPass(x0,W,b)
        delta = backwardPass(y0,a,W)
        dW, db = stochasticGradient(x0,a,delta)
        W = updateParameters(W,dW,eta)
        b = updateParameters(b,db,eta)
        newcost = costFunction(x,y,W,b)
        cost[i] = newcost
        result = checkResult(x,y,W,b)
        if stop==True:
            if result == True:
                print(i/10)
                break;
    return(W,b,cost)

def toMatrix(vector, dimensions):
    """Converts a vector to arrays of specified dimensions"""
    size = [0]
    matrix = []
    for d in dimensions:
        size.append(d[0]*d[1])
    split = np.cumsum(size)
    for i in range(len(dimensions)):
        v = vector[split[i]:split[i+1]]
        v = v.reshape(dimensions[i])
        matrix.append(v)
    return(matrix)

def toVector(arrays):
    """Converts list of arrays to a vector"""
    vector = []
    dimensions = []
    for i in range(len(arrays)):
        flatArray = arrays[i].flatten()
        vector = np.concatenate((vector,flatArray))
        dimensions.append(arrays[i].shape)
    return(vector,dimensions)

def updateParameters(par,gradient,eta):
    """Update parameter using gradient and step size"""
    update = []
    for i in range(len(par)):
        update.append(par[i]-eta*gradient[i])
    return(update)


"""
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

"""
