import numpy as np
import matplotlib.pyplot as plt
import Highams2018 as highams

full_hessian = highams.nnet.hessian

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