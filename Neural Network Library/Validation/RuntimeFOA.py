import torch
import numpy as np
import FirstOrderModel as fom
import TrainingAlgorithms as train
import timeit

# I was lazy and just copy pasted from the PyTorch Autograd file
# There is no reason to use PyTorch functions here

x_predictors = np.array([[0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7],
                        [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]]).T

y_outcomes = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]).T

x0 = torch.tensor(x_predictors, requires_grad=False, dtype=torch.float64)
w0 = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], requires_grad=True, dtype=torch.float64)
w1 = torch.tensor([[0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]], requires_grad=True, dtype=torch.float64)
w2 = torch.tensor([[1.6, 1.7], [1.8, 1.9], [2.0, 2.1], [2.2, 2.3]], requires_grad=True, dtype=torch.float64)

w0 = w0.detach().numpy()
w1 = w1.detach().numpy()
w2 = w2.detach().numpy()
weights = [w0,w1,w2]

start = timeit.default_timer()

y_predictions, weights, gradients = train.gradient_descent(x_predictors,
                                                  y_outcomes,
                                                  weights,
                                                  fom.first_order_model,
                                                  max_iterations=1)

stop = timeit.default_timer()

print('Time: ', stop - start)
print(gradients[0])