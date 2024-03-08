import matplotlib.pyplot as plt
import numpy as np

# Define the activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def leaky_relu(x, alpha=0.05):
    return np.where(x > 0, x, alpha * x)

def linear(x):
    return x

# Generate x values
x_values = np.linspace(-4, 4, 100)

# Calculate y values for ReLU, Sigmoid, Leaky ReLU, and Linear
y_relu = relu(x_values)
y_sigmoid = sigmoid(x_values)
y_leaky_relu = leaky_relu(x_values)
y_linear = linear(x_values)

# Create subplots with 2 rows and 2 columns
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot Sigmoid activation on the first subplot
axs[0, 0].plot(x_values, y_sigmoid, label='Sigmoid', color='red', linewidth=4)
axs[0, 0].set_title('Sigmoid Activation')
axs[0, 0].legend()

# Plot Linear activation on the second subplot
axs[0, 1].plot(x_values, y_linear, label='Linear', color='purple', linewidth=4)
axs[0, 1].set_title('Linear Activation')
axs[0, 1].legend()

# Plot ReLU activation on the third subplot
axs[1, 0].plot(x_values, y_relu, label='ReLU', color='blue', linewidth=4)
axs[1, 0].set_title('ReLU Activation')
axs[1, 0].legend()

# Plot Leaky ReLU activation on the fourth subplot
axs[1, 1].plot(x_values, y_leaky_relu, label='Leaky ReLU', color='green', linewidth=4)
axs[1, 1].set_title('Leaky ReLU Activation')
axs[1, 1].legend()

# Adjust layout for better spacing
plt.tight_layout()

plt.subplots_adjust(hspace=0.25)

# Show the plot
plt.show()
