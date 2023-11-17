import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

# Define the data
x_predictors = np.array([[0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7],
                         [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]]).T

y_outcomes = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]]).T

# Convert data to PyTorch tensors
x_train = torch.tensor(x_predictors, dtype=torch.float32)
y_train = torch.tensor(y_outcomes, dtype=torch.float32)

# Define the neural network model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # Input layer: 2 input features, 2 neurons in the first hidden layer
        self.fc2 = nn.Linear(2, 3)  # First hidden layer: 2 neurons, 3 neurons in the second hidden layer
        self.fc3 = nn.Linear(3, 2)  # Second hidden layer: 3 neurons, 2 output neurons

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Initialize the model
model = NeuralNetwork()

# Define loss function (Sum of Squared Errors) and optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.25)  # Stochastic Gradient Descent optimizer

def sse_loss(predictions, targets):
    return torch.sum((predictions - targets) ** 2)/2

# Training loop
iterations = 4000
batch_size = len(x_train)  # Use the full dataset as a single batch
loss_values = []

for iteration in range(iterations):
    # Forward pass
    outputs = model(x_train)
    loss = sse_loss(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_values.append(loss.item())


'''CODE FOR GENERATING THE PLOTS'''

# Scatterplot and Loss function plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Scatterplot
x_min, x_max = 0, 1
y_min, y_max = 0, 1

# Separate the data into two sets based on Y values
x_y0 = x_predictors[(y_outcomes[:, 0] == 1) & (y_outcomes[:, 1] == 0)]
x_y1 = x_predictors[(y_outcomes[:, 0] == 0) & (y_outcomes[:, 1] == 1)]

# Evaluate the trained model on the grid and plot the partitioned region
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
model.eval()
with torch.no_grad():
    predicted = model(grid_tensor).detach().numpy()

predicted_classes = np.argmax(predicted, axis=1)

# Create a scatterplot for Y=0 points and Y=1 points
ax1.scatter(x_y0[:, 0], x_y0[:, 1], c='bisque', label='Predicted Failure', marker='s')
ax1.scatter(x_y1[:, 0], x_y1[:, 1], c='palegreen', label='Predicted Success', marker='s')
ax1.scatter(xx.ravel(), yy.ravel(), c=np.where(predicted_classes == 0, 'bisque', 'palegreen'))
ax1.scatter(x_y0[:, 0], x_y0[:, 1], label='Failure', marker='x', s=75, color='black')
ax1.scatter(x_y1[:, 0], x_y1[:, 1], color='black', label='Success', marker='o', s=75)
ax1.set_xlabel('X1 - Axis')
ax1.set_ylabel('X2 - Axis')
ax1.set_title('Partitioned Region Resulting from Trained Model')
ax1.legend(loc=2)
ax1.grid(True)

# Loss function plot
ax2.plot(range(iterations), loss_values, color='red', label='Loss')
ax2.set_xlabel('Iterations')
ax2.set_ylabel('Loss')
ax2.set_title('Loss Function Progression')
ax2.legend()
ax2.grid(True)

plt.tight_layout()

# Display both plots side-by-side
plt.show()
