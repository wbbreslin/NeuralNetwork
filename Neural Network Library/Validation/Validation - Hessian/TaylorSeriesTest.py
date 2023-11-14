import SOA_Hessian as data
import Base as base
import numpy as np
import FirstOrderModel as fom

hessian = data.full_hessian
nnet = data.nnet

g0 = base.to_vector(data.nnet['Gradients'][0])
g1 = base.to_vector(data.nnet['Gradients'][1])
g2 = base.to_vector(data.nnet['Gradients'][2])
gradient = np.vstack((g0,g1,g2))

np.random.seed(100)
V0 = np.random.rand(3,2)
V1 = np.random.rand(3,3)
V2 = np.random.rand(4,2)
v0 = base.to_vector(V0)
v1 = base.to_vector(V1)
v2 = base.to_vector(V2)
epsilon = 10**-8

# Calculate J(W)
y_outcomes = nnet['Outcomes']
y_predictions = fom.forward_pass(nnet)['States'][-1]
cost = base.mean_squared_error(y_predictions, y_outcomes)

# Calculate J(W+eV)
nnet['Weights'][0] = nnet['Weights'][0] + V0 * epsilon
nnet['Weights'][1] = nnet['Weights'][1] + V1 * epsilon
nnet['Weights'][2] = nnet['Weights'][2] + V2 * epsilon
y_predictions = fom.forward_pass(nnet)['States'][-1]
cost_epsilon = base.mean_squared_error(y_predictions, y_outcomes)

# Calculate gradient : eV
ip0 = g0.T @ v0 * epsilon
ip1 = g1.T @ v1 * epsilon
ip2 = g2.T @ v2 * epsilon
ip = ip0 + ip1 + ip2

numerator = cost_epsilon - cost - ip

v = np.vstack((v0, v1, v2))
denominator = epsilon**2/2 * v.T @ hessian @ v

ratio = numerator / denominator
print(ratio)