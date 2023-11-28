import Highams2019_Train_GradientDescent as data
import torch
import torch.nn as nn
import numpy as np
import Base as base

nnet = data.nnet

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

def forward(x0):
    z0 = new_column(x0)
    x1 = torch.mm(z0, w0)
    x1 = torch.sigmoid(x1)
    z1 = new_column(x1)
    x2 = torch.mm(z1, w1)
    x2 = torch.sigmoid(x2)
    z2 = new_column(x2)
    x3 = torch.mm(z2, w2)
    x3 = torch.sigmoid(x3)
    return(x3)

x0 = torch.tensor(nnet["Predictors"], requires_grad=False, dtype=torch.float64)
w0 = torch.tensor(nnet["Weights"][0], requires_grad=True, dtype=torch.float64)
w1 = torch.tensor(nnet["Weights"][1], requires_grad=True, dtype=torch.float64)
w2 = torch.tensor(nnet["Weights"][2], requires_grad=True, dtype=torch.float64)
weights = [w0,w1,w2]

for dw in weights:
    v_length = dw.shape[0]*dw.shape[1]
    for df in weights:
        for i in range(v_length):
            x3 = forward(x0)
            criterion = CustomLoss()
            y_target = torch.tensor(nnet["Outcomes"], dtype=torch.float64)
            loss = criterion(x3, y_target)
            grad_loss_df = torch.autograd.grad(loss, df, create_graph=True)[0]
            vector_v = np.zeros((v_length,1))
            vector_v[i] = 1
            vector_v = base.to_matrix(vector_v, dw.shape)
            vector_v = torch.tensor(vector_v, requires_grad=False, dtype=torch.float64)
            hessian_vector_product = torch.autograd.grad(grad_loss_df, dw, vector_v, retain_graph=True)
            print("Hessian-vector product:", hessian_vector_product)