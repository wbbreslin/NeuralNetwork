import NNet as model
import numpy as np

'''The data set'''
x = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],
              [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]]).T

y= np.array([[1,1,1,1,1,0,0,0,0,0],
             [0,0,0,0,0,1,1,1,1,1]]).T


'''Define specifications for neural network'''
layers = [2,2,3,2]
activations = ['relu', 'sigmoid']
cost = 'MSE'

nnet = model.neural_network(x,y,layers,activations,cost)
predictions = nnet.predict(x)
nnet.forward()

print(nnet.weights[0])
print(nnet.augmented_weights[0])


