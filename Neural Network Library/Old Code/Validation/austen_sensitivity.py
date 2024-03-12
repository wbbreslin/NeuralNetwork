from __future__ import print_function
from matplotlib.colors import LogNorm
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.autograd.functional import vhp, jacobian
from matplotlib import pyplot as plt
from evdev import InputDevice, categorize, ecodes
import asyncio
import threading
import numpy as np

class Net(nn.Module):
    def __init__(self, data, hot_target):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 3)
        self.fc3 = nn.Linear(3, 2)
        self.data = data 
        self.hot_target = hot_target.float()

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        output = F.sigmoid(x)
        return output



def train_step(model, data, target, optimizer):
    optimizer.zero_grad()
    output = model(data)
    loss = F.mse_loss(output, target.float(), reduction='sum') / 2.0
    loss.backward()
    optimizer.step()
    return loss


def test(model, data, target):
    with torch.no_grad():
        output = model(data)
        test_loss = F.mse_loss(output, target.float(), reduction='sum') / 2.0
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        pred = F.one_hot(pred, num_classes=2)
        correct = pred.eq(target.view_as(pred)).sum().item() / 2

    return test_loss, correct

def train_model(x, y, x_validation, y_validation):
    model = Net(x,y)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    i = 0
    while i < 4000:
        train_loss = train_step(model, x, y, optimizer)
        i += 1

        if i % 1000 == 0:
            test_loss, correct = test(model, x_validation, y_validation)
            print(f'train loss: {train_loss:.2e}\ttest loss: {test_loss:.2e}\tcorrect: {correct} / {x_validation.shape[0]}')
    return model

def main():
    x = torch.tensor([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],
              [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]]).T

    y = torch.tensor([[1,1,1,1,1,0,0,0,0,0],
              [0,0,0,0,0,1,1,1,1,1]]).T

    x_validation = torch.tensor([[0.7,0.2,0.6,0.9],
                         [0.9,0.7,0.1,0.8]]).T

    y_validation = torch.tensor([[1,1,0,0],
                         [0,0,1,1]]).T

    model = train_model(x, y, x_validation, y_validation)

    def q(s):
        qoi = 0.5 * torch.linalg.norm(s * (model(x) - y))**2
        return qoi

    s = torch.ones(x.shape[0], 1)
    fso = jacobian(q, s)

    #ose = np.zeros(x.shape[0])
    #for i in range(x.shape[0]):
        #model = train_model(torch.cat([x[0:i], x[i+1:]]), torch.cat([y[0:i], y[i+1:]]), x_validation, y_validation)
        #qoi =  0.5 * torch.linalg.norm(model(x_validation) - y_validation)**2
        #ose[i] = qoi

    plt.plot(fso, label="fso")
    #plt.plot(ose, label="ose")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
