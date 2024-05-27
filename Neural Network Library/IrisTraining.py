import numpy as np
import pandas as pd
from Data import data
import NeuralNetwork as nn
import ActivationFunctions as af
import CostFunctions as cf
import matplotlib.pyplot as plt
import Base
import copy
import pickle

np.random.seed(314)
'''Import and preview the data'''
file_path = 'iris.csv'
df = pd.read_csv(file_path)
print(df.head())

'''Data cleaning'''
x_columns = df.iloc[:,:4]
y_columns = df['species']
x = x_columns.to_numpy()
y = y_columns.to_numpy()
unique_species, indices = np.unique(y, return_inverse=True)
unit_vectors = np.eye(len(unique_species))
y = unit_vectors[indices]
print("X dimensions:", x.shape)
print("Y dimensions:", y.shape)

iris = data(x,y)
training, validation = iris.test_train_split(train_percent=0.8)
training.x = training.x.T
training.y = training.y.T
validation.x = validation.x.T
validation.y = validation.y.T
print(training.x.shape)
print(validation.x.shape)

np.random.seed(200)
nnet = nn.neural_network(layers=[4,8,3],
                         activation_functions = [af.sigmoid, af.softmax],
                         cost_function=cf.half_SSE,
                         regularization=8*10**-1)


'''Train the model'''
itr1 = 40000
itr2 = 1000
step = 0.01
nnet.train(training, max_iterations =itr1, step_size=step, decay=0.9997) #0.9997


with open('trained_network.pkl', 'wb') as file:
    # Serialize and write the object to the file
    pickle.dump(nnet, file)

'''Plot costs'''
plt.plot(nnet.costs)
plt.show()

'''Evaluate Gradient and Hessian Norms'''
gradient = nnet.compute_gradient()
print("Gradient Norm", np.linalg.norm(nnet.gradient_vector))

'''Duplicate pretrained model for OSE'''
nnet_OSE_init = copy.deepcopy(nnet)

'''Duplicate trained model for FSO'''
nnet_FSO_J = copy.deepcopy(nnet)
nnet_FSO_q = copy.deepcopy(nnet)


'''FSO Sensitivity Analysis'''
nnet_FSO_J.forward(training)
nnet_FSO_J.backward(training)
nnet_FSO_J.backward_hyperparameter_derivative(training)
nnet_FSO_J.compute_hessian()
pdef = Base.is_positive_definite(nnet_FSO_J.hessian_matrix)
print("Positive Definite?:", pdef)
nnet_FSO_q.forward(validation)
nnet_FSO_q.backward(validation)
nnet_FSO_q.track_cost(validation)
nnet_FSO_q.compute_gradient()

eta = np.linalg.inv(nnet_FSO_J.hessian_matrix) @ nnet_FSO_q.gradient_vector
forecast_gradient = -nnet_FSO_J.dSW @ eta
sensitivity = -forecast_gradient


'''OSE Sensitivity Analysis'''
#Training - Compute Unmodified Validation Cost
nnet_OSE_validation = copy.deepcopy(nnet)
nnet_OSE_validation.forward(validation)
nnet_OSE_validation.track_cost(validation)
unmodified_cost = nnet_OSE_validation.costs[-1]
print("Original validation cost:", unmodified_cost)


cost_impact = []
perturbation = 1
for i in range(10):
    nnet_OSE = copy.deepcopy(nnet_OSE_init)
    training_OSE = copy.deepcopy(training)
    training_OSE.s = np.ones((120,1))
    training_OSE.s[i] = 1-perturbation
    nnet_OSE.train(training_OSE, max_iterations=itr2, step_size=step, decay=0.997)
    nnet_OSE.compute_gradient()
    norm = np.linalg.norm(nnet_OSE.gradient_vector)
    nnet_OSE_validation = copy.deepcopy(nnet_OSE)
    nnet_OSE_validation.forward(validation)
    nnet_OSE_validation.track_cost(validation)
    new_cost = nnet_OSE_validation.costs[-1]
    delta = new_cost-unmodified_cost
    cost_impact.append(delta)
    print("OSE relative validation cost (",i+1,"/",120,"):", new_cost, "gradient norm:", norm)

cost_impact = np.array(cost_impact).reshape(-1, 1)/perturbation



plt.plot(cost_impact, label="OSE")
plt.plot(sensitivity, label="FSO")
plt.legend()
plt.show()

x1 = training.x[0,:].reshape((120,1))
x2 = training.x[1,:].reshape((120,1))
x3 = training.x[2,:].reshape((120,1))
x4 = training.x[3,:].reshape((120,1))
y1 = training.y[0,:].reshape((120,1))
y2 = training.y[1,:].reshape((120,1))
y3 = training.y[2,:].reshape((120,1))

nnet.forward(training)
prediction = nnet.states[-1]
Pr1 = prediction.x[0,:].reshape((120,1))
Pr2 = prediction.x[1,:].reshape((120,1))
Pr3 = prediction.x[2,:].reshape((120,1))

OSE = cost_impact.reshape((120,1))
FSO = sensitivity.reshape((120,1))

df2 = np.hstack((x1,x2,x3,x4,y1,y2,y3,Pr1,Pr2,Pr3, FSO, OSE))
dataset = pd.DataFrame(df2, columns = ["X1", "X2","X3","X4",
                                       "Y1","Y2", "Y3",
                                       "Pr1", "Pr2", "Pr3",
                                       "FSO Impact",
                                       "OSE Impact"])

dataset.to_csv("iris_results_new.csv",index=False)