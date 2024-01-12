import Iris2class as data
import Base as base
import FirstOrderModel as fom
import copy
import numpy as np

x_test = data.x_test
y_test = data.y_test
nnet = base.load_nnet()
model = copy.deepcopy(nnet)
model['Pass Forward'] = x_test
model_output= fom.forward_pass(model)
predictions = np.round(model_output['States'][-1])
errors = y_test - predictions
print(errors)