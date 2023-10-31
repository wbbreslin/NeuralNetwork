import numpy as np
import Base as base
import TrainingAlgorithms as train
import SecondOrderModel as som
"""The data set of predictor variables"""
x_predictors = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],
                        [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]]).T

"""The data set of outcome variables"""
y_outcomes = np.array([[1,1,1,1,1,0,0,0,0,0],
                      [0,0,0,0,0,1,1,1,1,1]]).T

"""Define the neural network structure"""
np.random.seed(100)
neurons = np.array([2,2,3,2])
activations = ["sigmoid","sigmoid","sigmoid","sigmoid"]
weights, biases = base.create_network(neurons)
weights = base.augment_network(weights, biases)

nnet = {'Predictors': x_predictors,
        'Outcomes': y_outcomes,
        'Weights': weights,
        'Neurons': neurons}

nnet = train.gradient_descent(nnet,max_iterations=10**0)
print(nnet['Weights'][0])
print(nnet['Augmented_Weights'][0])


vectors = nnet['Gradients'].copy()
for i in range(len(vectors)):
    print(nnet['Gradients'][i].shape)
    vectors[i], dims = base.to_vector(vectors[i])

KTensors = []
for i in range(len(nnet['Weights'])):
    KT = som.Kron_Tensors(nnet['Weights'][i],10)
    KTensors.append(KT)


nnet = som.forward_pass(nnet,vectors)
nnet = som.backward_pass(nnet, vectors, KTensors)
print(nnet['Omegas'])
#print(np.round(nnet['States'][-1],3))
#18.413428542000474

#print(base.sigmoid_derivative(x_predictors).shape)