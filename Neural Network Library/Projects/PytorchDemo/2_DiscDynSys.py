import torch

def dynamical_system(x_0, iterations):
    current_value = x_0
    for i in range(iterations):
        next_value = current_value**2
        print(f"Iteration {i + 1}: {next_value.item()}")
        current_value = next_value
    return current_value

# Define initial value and number of iterations
initial_value = torch.tensor(2.0, requires_grad=True)
iterations = 3

final_result = dynamical_system(initial_value, iterations)
final_result.backward()

gradient_initial_value = initial_value.grad

print(f"The gradient with respect to the initial condition x0 = {initial_value.item()} is: {gradient_initial_value.item()}")