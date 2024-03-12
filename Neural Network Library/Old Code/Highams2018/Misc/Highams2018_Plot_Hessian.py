import numpy as np
import matplotlib.pyplot as plt
import Highams2018 as highams

nnet = highams.nnet
nnet.compute_hessian()
full_hessian = nnet.hessian_matrix

min = np.abs(np.min(full_hessian))
max = np.abs(np.max(full_hessian))
bound = np.max((min, max))

"""Plot the Hessian"""
plt.imshow(full_hessian, cmap='seismic', vmin=-bound, vmax=bound)
plt.colorbar()
plt.xlabel('Weight Parameter ID (23 parameters)')
plt.ylabel('Weight Parameter ID (23 parameters)')
plt.title('Hessian Matrix: ' + str(8000) + ' Iterations')
plt.show()

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

pd = is_pos_def(full_hessian)
print(pd)