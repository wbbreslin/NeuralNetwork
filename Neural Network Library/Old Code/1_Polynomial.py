import torch

def polynomial(x):
    return x**2 - 3*x + 2

# Function to compute the derivative of the polynomial
def derivative(x):
    x = torch.tensor(x, dtype=torch.float32, requires_grad=True) # Convert to torch tensor
    y = polynomial(x) # The forward model
    y.backward() # The backward (adjoint) model
    return x.grad

# Value at which you want to evaluate the derivative
x_value = 2

# Computing the derivative at x = x_value
derivative_value = derivative(x_value)
print(f"The derivative of the polynomial at x = {x_value} is: {derivative_value.item()}")
