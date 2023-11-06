import numpy as np
import matplotlib.pyplot as plt
import FDM_Hessian as data

'''Remove zeros from heatmap'''
full_hessian = data.full_hessian
full_hessian[full_hessian==0] = np.nan


"""Plot the Hessian"""
plt.imshow(full_hessian, cmap='viridis')
plt.colorbar()
plt.xlabel('Weight Parameter ID (23 parameters)')
plt.ylabel('Weight Parameter ID (23 parameters)')
plt.title('Hessian Matrix: ' + str(data.training_itr) + ' Iterations')
plt.show()