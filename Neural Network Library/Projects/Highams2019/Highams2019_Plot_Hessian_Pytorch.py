import Highams2019_Hessian_Pytorch as data
import matplotlib.pyplot as plt
import numpy as np

full_hessian = data.hessian_matrix
full_hessian = full_hessian.numpy()

min = np.abs(np.min(full_hessian))
max = np.abs(np.max(full_hessian))
bound = np.max((min, max))

"""Plot the Hessian"""
plt.imshow(full_hessian, cmap='seismic', vmin=-bound, vmax=bound)
plt.colorbar()
plt.xlabel('Weight Parameter ID (23 parameters)')
plt.ylabel('Weight Parameter ID (23 parameters)')
plt.title('Hessian Matrix: ' + str(4000) + ' Iterations')
plt.show()