import numpy as np
import matplotlib.pyplot as plt
import Highams2019_Hessian_SOA as data

full_hessian = data.full_hessian
inverse = np.linalg.inv(full_hessian)

min = np.abs(np.min(inverse))
max = np.abs(np.max(inverse))
bound = np.max((min, max))

"""Plot the Hessian"""
plt.imshow(inverse, cmap='seismic', vmin=-bound, vmax=bound)
plt.colorbar()
plt.xlabel('Weight Parameter ID (23 parameters)')
plt.ylabel('Weight Parameter ID (23 parameters)')
plt.title('Inverse of Hessian Matrix: ' + str(4000) + ' Iterations')
plt.show()
