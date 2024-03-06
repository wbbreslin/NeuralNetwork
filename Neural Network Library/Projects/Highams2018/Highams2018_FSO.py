import numpy as np
import Highams2018 as hig

nnet = hig.nnet
nnet_validation = hig.nnet_copy
training = hig.training
validation = hig.validation

# Sensitivity Analysis
nnet_validation.forward(validation)
nnet_validation.backward(validation)
nnet.compute_hessian()
nnet_validation.compute_gradient()
eta = np.linalg.inv(nnet.hessian_matrix) @ nnet_validation.gradient_vector
nnet.backward_hyperparameter_derivative(training) #nabla SW of J

n = training.x.shape[0]
array = list(range(n**2))
index = [array[i] for i in range(len(array)) if i % (n+1) == 0]

total_dJ = np.hstack(nnet.dJ)
total_dJ = total_dJ[index,:]

sensitivity = -total_dJ @ eta
