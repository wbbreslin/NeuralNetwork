import Highams2019_Train_GradientDescent as data1
import Highams2019_Train_StochasticGradient as data2
import matplotlib.pyplot as plt
import numpy as np

nnet_regular = data1.nnet
nnet_stochastic = data2.nnet

list1 = nnet_regular['Cost']  # Your first list with 4000 numbers
list2 = nnet_stochastic['Cost']  # Your second list with 8000 numbers

# Generate x-axis values for each list
x_values_list1 = np.arange(len(list1))
x_values_list2 = np.arange(len(list2))

# Create subplots with 1 row and 2 columns
fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# Plot the first list on the first subplot
axs[0].plot(x_values_list1, list1, color='b', label='List 1')
axs[0].set_xlabel('Iterations (n=10)')
axs[0].set_ylabel('Cost')
axs[0].set_title('Gradient Descent')

# Plot the second list on the second subplot
axs[1].plot(x_values_list2, list2, color='r', label='List 2')
axs[1].set_xlabel('Iterations (n=5)')
axs[1].set_ylabel('Cost')
axs[1].set_title('Stochastic Gradient')

fig.suptitle('Convergence of Cost Function')

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()