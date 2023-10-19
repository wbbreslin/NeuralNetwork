import torch
import torch.nn as nn
import numpy as np
import FirstOrderModel as fom
import TrainingAlgorithms as train
def new_column(input_tensor):
    num_rows = input_tensor.size(0)
    ones_column = torch.ones(num_rows, 1)
    tensor_with_ones = torch.cat((ones_column, input_tensor), dim=1)
    return tensor_with_ones

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, predicted, target):
        # Implement your custom loss calculation here
        diff = (predicted - target)
        loss = torch.trace(torch.mm(diff, diff.t()))/2

        return loss
# Define the data
x_predictors = np.array([[0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7],
                        [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]]).T

y_outcomes = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]).T

x0 = torch.tensor(x_predictors, requires_grad=False, dtype=torch.float64)
w0 = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], requires_grad=True, dtype=torch.float64)
w1 = torch.tensor([[0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [1.3, 1.4, 1.5]], requires_grad=True, dtype=torch.float64)
w2 = torch.tensor([[1.6, 1.7], [1.8, 1.9], [2.0, 2.1], [2.2, 2.3]], requires_grad=True, dtype=torch.float64)

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

# Create target tensor (replace with your actual target)
y_target = torch.tensor(y_outcomes, dtype=torch.float64)

# Calculate the loss
loss = criterion(x3, y_target)

# Compute gradients
loss.backward()

# Access the gradients of w0, w1, and w2
grad_w0 = w0.grad
grad_w1 = w1.grad
grad_w2 = w2.grad

print("Gradient of w0:")
print(grad_w0)

print("Gradient of w1:")
print(grad_w1*100)

print("Gradient of w2:")
print(grad_w2)

w0 = w0.detach().numpy()
w1 = w1.detach().numpy()
w2 = w2.detach().numpy()
weights = [w0,w1,w2]

y_predictions, weights, gradients = train.gradient_descent2(x_predictors,
                                                  y_outcomes,
                                                  weights,
                                                  fom.first_order_model2,
                                                  max_iterations=1)

g0_auto = grad_w0.numpy()
g1_auto = grad_w1.numpy()
g2_auto = grad_w2.numpy()

g0_math = gradients[0]
g1_math = gradients[1]
g2_math = gradients[2]

d0 = g0_auto-g0_math
d1 = g1_auto-g1_math
d2 = g2_auto-g2_math

print(d0)
print(d1)
print(d2)
