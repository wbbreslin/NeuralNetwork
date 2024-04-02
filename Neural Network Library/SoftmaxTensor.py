import numpy as np
import ActivationFunctions as af

x = np.array([[0.1, 0.3, 0.6],
              [0.1, 0.6, 0.2],
              [0.1, 0.1, 0.4]])

print("Softmax Hessian")
H = af.softmax_second_derivative(x)
print(H[0,:,:,:])
