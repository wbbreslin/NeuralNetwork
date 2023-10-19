# Import necessary libraries (including PyTorch)
import torch
import torch.nn as nn
import torch.optim as optim

# Set the seed for the CPU
seed = 42
torch.manual_seed(seed)

# Set the seed for CUDA (if available and you're using GPU)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Ensure that all operations are deterministic on the CPU and GPU (if available)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(2, 2),  # Input layer (2 input features, 2 hidden units)
            nn.Sigmoid()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(2, 3),  # Hidden layer (2 hidden units, 3 hidden units)
            nn.Sigmoid()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(3, 2),  # Output layer (3 hidden units, 2 output units)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# Create an instance of the neural network
model = NeuralNetwork()

# Access and print the initial weight matrices and bias vectors
initial_weight_layer1 = model.layer1[0].weight
initial_bias_layer1 = model.layer1[0].bias

initial_weight_layer2 = model.layer2[0].weight
initial_bias_layer2 = model.layer2[0].bias

initial_weight_layer3 = model.layer3[0].weight
initial_bias_layer3 = model.layer3[0].bias

# # Print the initial weight matrices and bias vectors
# print("Initial weight matrix for layer 1:")
# print(initial_weight_layer1)
#
# print("Initial bias vector for layer 1:")
# print(initial_bias_layer1)
#
# print("Initial weight matrix for layer 2:")
# print(initial_weight_layer2)
#
# print("Initial bias vector for layer 2:")
# print(initial_bias_layer2)
#
# print("Initial weight matrix for layer 3:")
# print(initial_weight_layer3)
#
# print("Initial bias vector for layer 3:")
# print(initial_bias_layer3)

# Define a loss function (e.g., mean squared error) and an optimizer (e.g., regular gradient descent)
criterion = nn.MSELoss()
learning_rate = 0.25
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Generate example input data and target output data (replace with your actual data)
input_data = torch.tensor([[0.1, 0.1],
                           [0.3, 0.4],
                           [0.1, 0.5],
                           [0.6, 0.9],
                           [0.4, 0.2],
                           [0.6, 0.3],
                           [0.5, 0.6],
                           [0.9, 0.2],
                           [0.4, 0.4],
                           [0.7, 0.6]], dtype=torch.float32)

target_output = torch.tensor([[1.0, 0.0],
                              [1.0, 0.0],
                              [1.0, 0.0],
                              [1.0, 0.0],
                              [1.0, 0.0],
                              [0.0, 1.0],
                              [0.0, 1.0],
                              [0.0, 1.0],
                              [0.0, 1.0],
                              [0.0, 1.0]], dtype=torch.float32)

# Training loop with batch gradient descent
num_epochs = 1000
batch_size = len(input_data)  # Batch size equal to the dataset size

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(input_data)

    # Compute the loss
    loss = criterion(outputs, target_output)

    # Backpropagation and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) == 1:
        print(loss.item())

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')


# After training, you can use the trained model for inference.

# Access the final weight matrices for each layer
final_weight_layer1 = model.layer1[0].weight.detach().numpy()
final_weight_layer2 = model.layer2[0].weight.detach().numpy()
final_weight_layer3 = model.layer3[0].weight.detach().numpy()

# Print the final weight matrices



#
# print("Final weight matrix for layer 3:")
# print(final_weight_layer3)
#


# print("Final weight matrix for layer 2:")
# print(final_weight_layer2)

print("Final weight matrix for layer 1:")
print(final_weight_layer1)

for name, param in model.named_parameters():
    if param.grad is not None:
        print(f'Gradient for {name}:')
        print(param.grad)