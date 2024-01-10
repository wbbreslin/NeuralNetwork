import torch
import Highams2019_Train_GradientDescent as data
import numpy as np
import Base as base
import matplotlib.pyplot as plt

# Function to add a column of ones to a tensor
def new_column(input_tensor):
    num_rows = input_tensor.size(0)
    ones_column = torch.ones(num_rows, 1)
    tensor_with_ones = torch.cat((ones_column, input_tensor), dim=1)
    return tensor_with_ones

nnet = data.nnet

# Example: Load your network weights (Replace with your initialization)
x0 = torch.tensor(nnet["Predictors"], requires_grad=False, dtype=torch.float64)
w0 = torch.tensor(nnet["Weights"][0], requires_grad=True, dtype=torch.float64)
w1 = torch.tensor(nnet["Weights"][1], requires_grad=True, dtype=torch.float64)
w2 = torch.tensor(nnet["Weights"][2], requires_grad=True, dtype=torch.float64)

# Example: Define your custom loss function
def custom_loss(predicted, target):
    diff = (predicted - target)
    loss = torch.trace(torch.mm(diff, diff.t())) / 2
    return loss

# Function to perform forward pass through the network
def forward(x0, w0, w1, w2):
    z0 = new_column(x0)
    x1 = torch.mm(z0, w0)
    x1 = torch.sigmoid(x1)
    z1 = new_column(x1)
    x2 = torch.mm(z1, w1)
    x2 = torch.sigmoid(x2)
    z2 = new_column(x2)
    x3 = torch.mm(z2, w2)
    x3 = torch.sigmoid(x3)
    return x3

# Calculate the loss
output = forward(x0, w0, w1, w2)
y_target = torch.tensor(nnet["Outcomes"], dtype=torch.float64)
loss = custom_loss(output, y_target)

# Compute gradients
params = [w0, w1, w2]
grads = torch.autograd.grad(loss, params, create_graph=True)

# Calculate the Hessian matrix
hessian = []
for grad_elem in grads:
    grad_flat = grad_elem.view(-1)
    hess_elem = []
    for grad_i in grad_flat:
        # Compute second derivatives
        hess_grad = torch.autograd.grad(grad_i, params, retain_graph=True)
        hess_flat = [h.view(-1) for h in hess_grad]
        hess_elem.append(torch.cat(hess_flat))
    hessian.append(torch.stack(hess_elem))

hessian_matrix = torch.cat(hessian)

V = np.array([1, 4, 2, 5, 3, 6, 7, 10, 13, 8, 11, 14, 9, 12, 15, 16, 20, 17, 21, 18, 22, 19, 23])-1
U = np.arange(23)
permutation = torch.tensor(base.permutation_matrix_by_indices(U,V), requires_grad=False, dtype=torch.float64)

hessian_matrix = torch.mm(permutation.T, hessian_matrix)
hessian_matrix = torch.mm(hessian_matrix, permutation)

print("Hessian matrix shape:", hessian_matrix.shape)
print(hessian_matrix[:,0])
print(sum(hessian_matrix[:,0]))
