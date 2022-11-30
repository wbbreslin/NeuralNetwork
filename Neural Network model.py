import numpy as np
import matplotlib.pyplot as plt
from itertools import product

#Creating and formatting the data set
x1 = np.array([0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7])
x2 = np.array([0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6])
y1 = np.hstack([np.ones(5), np.zeros(5)])
y2 = np.hstack([np.zeros(5), np.ones(5)])
y = np.vstack([y1,y2])
xt = np.transpose(np.vstack([x1,x2]))
yt = np.transpose(y)

def activate(x,W,b):
    z = W@x+b
    y = 1+np.exp(-z)
    y = 1/y
    return(y)

def backwardPass(y,a,W):
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

def cost(x,y,W,b):
    """The cost function to be minimized"""
    n = x.shape[0]
    costvec = np.zeros(n)
    for i in range(n):
        x0 = np.array([[x[i,0]],
                       [x[i,1]]])
        y0 = np.array([[y[i,0]],
                       [y[i,1]]])
        a = forwardPass(x0,W,b)
        costvec[i] = np.linalg.norm(y0-a[-1])**2
    cost = sum(costvec)/10
    return(cost)

def createNetwork(neurons):
    """Initiates Network parameters given a list of neurons per layer"""
    W = []; b = []
    for i in range(len(neurons)-1):
        W.append(0.5*np.random.rand(neurons[i+1],neurons[i]))
        b.append(0.5*np.random.rand(neurons[i+1],1))
    return(W,b)

def forwardPass(x,W,b):
    """Passes a single data point through the network"""
    a = []
    for i in range(len(W)):
        if i == 0:
            a.append(activate(x,W[i],b[i]))
        else:
            a.append(activate(a[i-1],W[i],b[i]))
    return(a)

def gradient(x,a,delta):
    layers = len(delta)
    wGradient = []
    bGradient = delta
    for i in range(layers):
        if i==0:
            wGradient.append(delta[i]@np.transpose(x))
        else:
            wGradient.append(delta[i]@np.transpose(a[i-1]))
    return(wGradient,bGradient)

def plotCost():
    """Plot the log-cost against the iterates"""
    plt.plot(np.array(range(len(savecost))),np.log10(savecost))
    plt.title("Convergence of Cost Function")
    plt.xlabel("Iteration")
    plt.ylabel("Log Cost")
    plt.show()

def plotRegion():
    """Plot of the predicted region partition"""
    xh = np.arange(0,1,0.001)
    xv = np.arange(0,1,0.001)
    xall = np.array(list(product(xh, xv)))
    ySave = xall*0
    for i in range(xall.shape[0]):
        xp = np.array([[xall[i,0]],
                      [xall[i,1]]])
        a = forwardPass(xp,W,b)
        ySave[i,0]=a[-1][0]
        ySave[i,1]=a[1][1]
    ySave = np.around(ySave,0)
    yc = ySave[:,0]
    plt.scatter(xall[:,0]*yc,xall[:,1]*yc)
    plt.show()
    
def randomXY(x,y):
    """This is redundant, just select random k"""
    n = x.shape[0]
    k = np.random.randint(10)
    x0 = np.array([[x[k,0]],
                   [x[k,1]]])
    y0 = np.array([[y[k,0]],
                   [y[k,1]]])
    return(x0,y0)

def selectXY(x,y,k):
    """Selects an (x,y) pair based on index k as an np.array"""
    n = x.shape[0]
    x0 = np.array([[x[k,0]],
                   [x[k,1]]])
    y0 = np.array([[y[k,0]],
                   [y[k,1]]])
    return(x0,y0)

def toVector(arrays):
    """Converts list of arrays to a vector"""
    vector = []
    dimensions = []
    for i in range(len(arrays)):
        flatArray = arrays[i].flatten()
        vector = np.concatenate((vector,flatArray))
        dimensions.append(arrays[i].shape)
    return(vector,dimensions)

def toMatrix(vector, dimensions):
    """Converts a vector to arrays of specified dimensions"""
    size = [0]
    matrix = []
    for d in dimensions:
        size.append(d[0]*d[1])
    split = np.cumsum(size)
    for i in range(len(dimensions)):
        v = vector[split[i]:split[i+1]]
        d = dimensions[i]
        v = v.reshape(dimensions[i])
        matrix.append(v)
    return(matrix)
    
def updateParameters(par,gradient,eta):
    """Update parameter using gradient and step size"""
    update = []
    for i in range(len(par)):
        update.append(par[i]-eta*gradient[i])
    return(update)

eta = 0.05 #learning rate
itr = 10**5 #number of iterations
savecost = np.zeros(itr) #monitor cost function

W,b = createNetwork([2,2,3,2])     
for i in range(itr):
    for k in range(xt.shape[0]):
        x0, y0 = selectXY(xt,yt,k)
        a = forwardPass(x0,W,b)
        delta = backwardPass(y0,a,W)
        wGradient, bGradient = gradient(x0,a,delta)
        W = updateParameters(W,wGradient,eta)
        b = updateParameters(b,bGradient,eta)
    newcost = cost(xt,yt,W,b)
    savecost[i] = newcost
