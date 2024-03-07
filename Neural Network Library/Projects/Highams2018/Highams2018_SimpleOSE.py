import Highams2018 as hg
from Data import data
import numpy as np

nnet = hg.nnet
nnet_OSE = hg.nnet_OSE_ref
training = hg.training
validation = hg.validation

i=1
x_OSE = np.delete(training.x,i,axis=0)
y_OSE = np.delete(training.y,i,axis=0)
training = data(x_OSE,y_OSE)

nnet_OSE.train(training, max_iterations = 12000, step_size=0.25)

nnet.forward(validation)
nnet.backward(validation)
nnet.track_cost(validation)

nnet_OSE.forward(validation)
nnet_OSE.backward(validation)
nnet_OSE.track_cost(validation)

print(nnet.costs[-1])
print(nnet_OSE.costs[-1])
delta = nnet_OSE.costs[-1]-nnet.costs[-1]
print(delta)