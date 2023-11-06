import numpy as np
import Base as base
import TrainingAlgorithms as train

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


#nnet = train.gradient_descent(nnet,max_iterations=10**0)

# Generate vectors of appropriate dimensions for testing purposes
#vectors = nnet['Gradients'].copy()
#for i in range(len(vectors)):
#    vectors[i], dims = base.to_vector(vectors[i])
#    print(vectors[i].shape)

'''
nnet = som.forward_pass(nnet,vectors)
nnet = som.backward_pass(nnet,vectors)
#bv = som.Bv_Tensor(nnet, vectors, 0)
#cv = som.Cv_Tensor(nnet, 2)
#print(cv)
#nnet = som.backward_pass(nnet, vectors, KTensors)
'''
