import torch
from torch.autograd import Variable

def hessian(model, loss, params):
    # Create an empty Hessian matrix
    H = torch.zeros(params.size(0), params.size(0))

    for i in range(params.size(0)):
        # Compute the second derivative with respect to the i-th parameter
        grad2 = torch.autograd.grad(loss, params[i], create_graph=True)[0]

        # Compute the diagonal of the Hessian matrix
        H[i, i] = torch.autograd.grad(grad2, params[i])[0]

        for j in range(i + 1, params.size(0)):
            # Compute the off-diagonal elements of the Hessian matrix
            grad2 = torch.autograd.grad(grad2, params[j])[0]
            H[i, j] = grad2
            H[j, i] = grad2

    return H

# Example usage:
x = torch.tensor([2.0], requires_grad=True)
y = x**3 + 3*x**2 - 2*x + 1  # Example loss function
params = [x]

# Compute the Hessian matrix
H = hessian(y, x, params)

print("Hessian matrix:")
print(H)