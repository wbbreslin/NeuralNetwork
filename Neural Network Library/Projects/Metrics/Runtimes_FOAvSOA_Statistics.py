import numpy as np
from NNET import neural_network
from Data import data
import ActivationFunctions as af
import CostFunctions as cf
import matplotlib.pyplot as plt
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

'''Time the FOA model'''
foa_time_data = []
soa_time_data = []
rel_time_data = []
for i in range(100):
    '''FOA Runtime'''
    start_time_foa = time.time()
    nnet.forward(training)
    nnet.backward(training)
    end_time_foa = time.time()
    foa_time = end_time_foa - start_time_foa
    foa_time_data.append(foa_time)
    '''SOA Runtime'''
    start_time_soa = time.time()
    nnet.soa_forward(vectors)
    nnet.soa_backward(vectors)
    end_time_soa = time.time()
    soa_time = end_time_soa - start_time_soa
    soa_time_data.append(soa_time)
    '''Relative Runtime'''
    rel_time = soa_time/foa_time
    rel_time_data.append(rel_time)

foa_avg = np.median(foa_time_data)
soa_avg = np.median(soa_time_data)
rel_avg = np.median(rel_time_data)


plt.plot(rel_time_data)
plt.show()

print("FOA runtime:", foa_avg)
print("SOA runtime:", soa_avg)
print("Relative Runtime Analysis")
print("Mean:", np.mean(rel_time_data))
print("Med.:", rel_avg)
print("Min.:", np.min(rel_time_data))
print("Max.:", np.max(rel_time_data))
