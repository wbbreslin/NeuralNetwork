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

'''
dfval = np.hstack((validation.x.T, validation.y.T))
val_export = pd.DataFrame(dfval, columns = ["X1", "X2","X3","X4","Y1","Y2", "Y3"])
val_export.to_csv("validation_data.csv",index=False)
'''


np.random.seed(200)

with open('trained_network.pkl', 'rb') as file:
    # Deserialize and load the object from the file
    nnet = pickle.load(file)

'''Evaluate Gradient and Hessian Norms'''
gradient = nnet.compute_gradient()
print("Gradient Norm", np.linalg.norm(nnet.gradient_vector))

'''Duplicate pretrained model for OSE'''
nnet_OSE_init = copy.deepcopy(nnet)

'''Duplicate trained model for FSO'''
nnet_FSO_J = copy.deepcopy(nnet)
nnet_FSO_q = copy.deepcopy(nnet)

def regularization_contribution(nnet):
    reg = [np.linalg.norm(w) ** 2 for w in nnet.weights]
    reg = np.sum(reg) / 2 * nnet.regularization
    return reg


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
weights_vec = [Base.to_vector(w) for w in nnet_FSO_q.weights]
weights_vec = np.vstack(weights_vec)
reg_grad = nnet_FSO_q.regularization * weights_vec
nnet_FSO_q.gradient_vector = nnet_FSO_q.gradient_vector - reg_grad


eta = np.linalg.inv(nnet_FSO_J.hessian_matrix) @ nnet_FSO_q.gradient_vector
forecast_gradient = -nnet_FSO_J.dSW @ eta
sensitivity = -forecast_gradient
print("First few sensitivities:", sensitivity[0:3])

'''OSE Sensitivity Analysis'''
#Training - Compute Unmodified Validation Cost
nnet_OSE_validation = copy.deepcopy(nnet)
nnet_OSE_validation.forward(validation)
nnet_OSE_validation.track_cost(validation)
unmodified_cost = nnet_OSE_validation.costs[-1]
reg_adjustment = regularization_contribution(nnet_OSE_validation) #New
unmodified_cost = unmodified_cost - reg_adjustment
print("Original validation cost:", unmodified_cost)

cost_impact = []
perturbation = 1
for i in range(120):
    nnet_OSE = copy.deepcopy(nnet_OSE_init)
    training_OSE = copy.deepcopy(training)
    training_OSE.s = np.ones((120,1))
    training_OSE.s[i] = 1-perturbation
    nnet_OSE.train(training_OSE, max_iterations=10000, step_size=0.01, decay=0.9997)
    nnet_OSE.compute_gradient()
    norm = np.linalg.norm(nnet_OSE.gradient_vector)
    nnet_OSE_validation = copy.deepcopy(nnet_OSE)
    nnet_OSE_validation.forward(validation)
    nnet_OSE_validation.track_cost(validation)
    reg_adjustment = regularization_contribution(nnet_OSE_validation) #New
    new_cost = nnet_OSE_validation.costs[-1]
    delta = new_cost-unmodified_cost
    delta = delta - reg_adjustment
    cost_impact.append(delta)
    print("OSE relative validation cost (",i+1,"/",120,"):", new_cost, "gradient norm:", norm)
    print("OSE:", delta, "FSO:", sensitivity[i])

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
Pr1 = prediction[0,:].reshape((120,1))
Pr2 = prediction[1,:].reshape((120,1))
Pr3 = prediction[2,:].reshape((120,1))

OSE = cost_impact.reshape((120,1))
FSO = sensitivity.reshape((120,1))

df2 = np.hstack((x1,x2,x3,x4,y1,y2,y3,Pr1,Pr2,Pr3, FSO, OSE))
dataset = pd.DataFrame(df2, columns = ["X1", "X2","X3","X4",
                                       "Y1","Y2", "Y3",
                                       "Pr1", "Pr2", "Pr3",
                                       "FSO Impact",
                                       "OSE Impact"])

dataset.to_csv("iris_results_new.csv",index=False)