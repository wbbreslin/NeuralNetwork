import torch.nn as nn
import numpy as np
import torch
import timeit


# Define the data
x_predictors = np.array([[0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7],
                        [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]]).T

y_outcomes = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]).T

x0 = torch.tensor(x_predictors, requires_grad=False, dtype=torch.float64)
w0 = torch.tensor(nnet["Weights"][0], requires_grad=True, dtype=torch.float64)
w1 = torch.tensor([[0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]], requires_grad=True, dtype=torch.float64)
w2 = torch.tensor([[1.6, 1.7], [1.8, 1.9], [2.0, 2.1], [2.2, 2.3]], requires_grad=True, dtype=torch.float64)

# Create target tensor
y_target = torch.tensor(y_outcomes, dtype=torch.float64)

start = timeit.default_timer()

for i in range(4000):
    z0 = new_column(x0)
    x1 = torch.mm(z0, w0)
    x1 = torch.sigmoid(x1)

    z1 = new_column(x1)
    x2 = torch.mm(z1, w1)
    x2 = torch.sigmoid(x2)

    z2 = new_column(x2)
    x3 = torch.mm(z2, w2)
    x3 = torch.sigmoid(x3)

    # Define a loss function (e.g., mean squared error)
    criterion = CustomLoss()

    # Calculate the loss
    loss = criterion(x3, y_target)

    # Compute gradients
    loss.backward()

stop = timeit.default_timer()

print('Time: ', stop - start)
print(w0.grad)