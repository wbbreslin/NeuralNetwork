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


# Initialize Weights and Biases
W2 = 0.5*np.random.rand(2,2)
W3 = 0.5*np.random.rand(3,2)
W4 = 0.5*np.random.rand(2,3)
b2 = 0.5*np.random.rand(2,1)
b3 = 0.5*np.random.rand(3,1)
b4 = 0.5*np.random.rand(2,1)
W=[W2,W3,W4]
b=[b2,b3,b4]


def activate(x,W,b):
    z = W@x+b
    y = 1+np.exp(-z)
    y = 1/y
    return(y)


def cost(x,y,W,b):
    n = x.shape[0]
    costvec = np.zeros(n)
    for i in range(n):
        x0 = np.array([[x[i,0]],
                       [x[i,1]]])
        y0 = np.array([[y[i,0]],
                       [y[i,1]]])
        a = forwardPass(x0,W,b)
        costvec[i] = np.linalg.norm(y0-a[-1])
    costval = np.linalg.norm(costvec)**2
    return(costval)


def randomXY(x,y):
    n = x.shape[0]
    k = np.random.randint(10)
    x0 = np.array([[x[k,0]],
                   [x[k,1]]])
    y0 = np.array([[y[k,0]],
                   [y[k,1]]])
    return(x0,y0)

def selectXY(x,y,k):
    n = x.shape[0]
    x0 = np.array([[x[k,0]],
                   [x[k,1]]])
    y0 = np.array([[y[k,0]],
                   [y[k,1]]])
    return(x0,y0)

def forwardPass(x,W,b):
    a = []
    for i in range(len(W)):
        if i == 0:
            a.append(activate(x,W[i],b[i]))
        else:
            a.append(activate(a[i-1],W[i],b[i]))
    return(a)


def backwardPass(y,a,W):
    layers = len(W)
    delta = []
    for i in reversed(range(layers)):
        j = layers-i-1
        d = np.diagflat(a[i]*(1-a[i]))
        if i == layers-1:
            delta.append(d@(a[i]-y))
        else:
            delta.append(d@(np.transpose(W[i+1])@delta[j-1]))
    delta.reverse()
    return(delta)

    
def updateGradient(x,eta,a,W,b):
    for i in range(len(W)):
        b[i] = b[i]-eta*delta[i]
        if i == 0:
            W[i] = W[i]-eta*delta[i]@np.transpose(x)
        else:
            W[i] = W[i]-eta*delta[i]@np.transpose(a[i-1])
    return(W,b)


eta = 0.05 #learning rate
itr = 10**5 #number of iterations
savecost = np.zeros(itr) #monitor cost function


for i in range(itr):
    for k in range(xt.shape[0]):
        x0, y0 = selectXY(xt,yt,k)
        a = forwardPass(x0,W,b)
        delta = backwardPass(y0,a,W)
        W, b = updateGradient(x0,eta,a,W,b)
    newcost = cost(xt,yt,W,b)
    savecost[i] = newcost


plt.plot(np.array(range(itr)),np.log10(savecost))
plt.title("Convergence of Cost Function")
plt.xlabel("Iteration")
plt.ylabel("Log Cost")
plt.show()

"""
xh = np.arange(0,1,0.01)
xv = np.arange(0,1,0.01)
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
"""
