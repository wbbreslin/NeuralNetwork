import Highams2019 as data
import torch.nn as nn
import torch

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

'''Import the neural network and the data set'''
nnet = data.nnet

x0 = torch.tensor(nnet["Predictors"], requires_grad=False, dtype=torch.float64)
w0 = torch.tensor(nnet["Weights"][0], requires_grad=True, dtype=torch.float64)
w1 = torch.tensor(nnet["Weights"][1], requires_grad=True, dtype=torch.float64)
w2 = torch.tensor(nnet["Weights"][2], requires_grad=True, dtype=torch.float64)

# Create target tensor
y_target = torch.tensor(nnet["Outcomes"], dtype=torch.float64)

learning_rate = 0.25  # Define your learning rate
optimizer = torch.optim.SGD([w0, w1, w2], lr=learning_rate)  # Use SGD optimizer

batch_size = len(x0)  # Set batch size equal to the full dataset size

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

    optimizer.zero_grad()

    # Compute gradients
    loss.backward()

    optimizer.step()

print(w0.grad)