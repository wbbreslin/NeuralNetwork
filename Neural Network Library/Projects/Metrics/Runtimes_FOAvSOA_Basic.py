import numpy as np
from NNET import neural_network
from Data import data
import ActivationFunctions as af
import CostFunctions as cf
import Base as base
import time



'''Data'''
x = np.array([[0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7],
              [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]]).T

y = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]).T


'''Define the network structure'''
np.random.seed(333)
nnet = neural_network(layers=[2, 2, 3, 2],
                      activation_functions=[af.sigmoid,
                                            af.sigmoid,
                                            af.sigmoid],
                      cost_function=cf.half_SSE)

training = data(x, y)


vectors = [np.ones_like(w) for w in nnet.weights]
vectors = [base.to_vector(v) for v in vectors]

'''FOA Runtime'''
start_time_foa = time.time()
nnet.forward(training)
nnet.backward(training)
end_time_foa = time.time()
foa_time = end_time_foa - start_time_foa

'''SOA Runtime'''
start_time_soa = time.time()
nnet.soa_forward(vectors)
nnet.soa_backward(vectors)
end_time_soa = time.time()
soa_time = end_time_soa - start_time_soa

'''Relative Runtime'''
rel_time = soa_time/foa_time

print("FOA Runtime:", foa_time)
print("SOA Runtime:", soa_time)
print("Relative Runtime:", rel_time)