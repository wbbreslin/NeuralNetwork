import numpy as np
import Base as base
import FirstOrderModel as fom
import TrainingAlgorithms as train

"""The data set of predictor variables"""
x_predictors = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],
                        [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]]).T

"""The data set of outcome variables"""
y_outcomes = np.array([[1,1,1,1,1,0,0,0,0,0],
                      [0,0,0,0,0,1,1,1,1,1]]).T

"""Define the neural network structure"""
#np.random.seed(100)
neurons = np.array([2,2,3,2])
activations = ["sigmoid","sigmoid","sigmoid","sigmoid"]
weights, biases = base.create_network(neurons)
augmented_weights = base.augment_network(weights, biases)
"""
states, augmented_states, first_derivatives = fom.forward_pass2(x_predictors,
                                                                augmented_weights)

Lambdas, gradients = fom.backward_pass2(states,
                                        y_outcomes,
                                        first_derivatives,
                                        augmented_weights)
"""

y_predictions, gradients = train.gradient_descent2(x_predictors,
                                                  y_outcomes,
                                                  augmented_weights,
                                                  fom.first_order_model2,
                                                  max_iterations=1000)

#print(np.round(y_predictions,3))
print(base.mean_squared_error(y_predictions, y_outcomes))
