import numpy as np
import Highams2018 as hig
import Base as base
import copy

nnet = hig.nnet_FSO_copy
nnet_validation = copy.deepcopy(nnet)
training = hig.training
validation = hig.validation

# Sensitivity Analysis
nnet.forward(training)
nnet.backward(training)
nnet.track_cost(training)
nnet_validation.forward(validation)
nnet_validation.backward(validation)
nnet.backward_hyperparameter_derivative(training)
nnet_validation.track_cost(validation)
#nnet_validation.update(step_size=0.25)
nnet.compute_hessian()
nnet_validation.compute_gradient()
eta = np.linalg.inv(nnet.hessian_matrix) @ nnet_validation.gradient_vector
sensitivity = nnet.dJ @ eta
print(sensitivity)

plt.plot(sensitivity)
plt.show()