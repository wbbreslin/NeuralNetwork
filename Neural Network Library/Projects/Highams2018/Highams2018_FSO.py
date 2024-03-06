import numpy as np
import Highams2018 as hig
import Base as base

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


sensitivity = nnet.dJ @ eta
print(sensitivity)
