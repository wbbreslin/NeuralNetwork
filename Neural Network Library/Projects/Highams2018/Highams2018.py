import numpy as np
from NNET import neural_network
from Data import data
import ActivationFunctions as af
import CostFunctions as cf

"""The data set of predictor variables"""
x = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],
              [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]]).T


"""The data set of outcome variables"""
y = np.array([[1,1,1,1,1,0,0,0,0,0],
              [0,0,0,0,0,1,1,1,1,1]]).T

#np.random.seed(100)

df = data(x,y)
training, validation = df.test_train_split(train_percent=1)
nnet = neural_network(layers=[2,2,3,2],
                         activation_functions = [af.sigmoid,
                                                 af.sigmoid,
                                                 af.sigmoid],
                         cost_function=cf.mean_squared_error)

nnet.randomize_weights()
nnet.train(df, max_iterations = 4000, step_size=0.25)
nnet.compute_hessian()
nnet.compute_gradient()


print(nnet.gradient_vector)
print(nnet.gradients[0])