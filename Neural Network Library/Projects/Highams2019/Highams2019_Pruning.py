import numpy as np
import Base as base
import matplotlib.pyplot as plt
import NetworkPruning as prune

"""The data set of predictor variables"""
x_predictors = np.array([[0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7],
                        [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6]]).T

"""The data set of outcome variables"""
y_outcomes = np.array([[1,1,1,1,1,0,0,0,0,0],
                      [0,0,0,0,0,1,1,1,1,1]]).T

"""Define the neural network structure"""
np.random.seed(100)
nnet = base.create_network(x_predictors,
                           y_outcomes,
                           neurons = [2,4,4,2],
                           activations = [base.sigmoid,
                                          base.sigmoid,
                                          base.sigmoid])


iterations = 31
nnet = prune.iterative_prune(nnet,itr=iterations, remove=1)

z = np.log(nnet['Cost'])

#xvline = [4000 * (i + 1) for i in range(iterations)]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlim((0,(iterations+1)*4000))
ax1.plot(z)
#ax1_ticks = list(range(0,(iterations+2)*4000,4000))
#ax1.set_xticks(ax1_ticks)
ax2 = ax1.twiny()
ax2.set_xlim((0,(iterations+1)))
ax2_ticks = list(range(0,(iterations+1),1))
ax2.set_xticks(ax2_ticks)

ax1.set_xlabel("Training Iterations")
ax2.set_xlabel("Pruning Iterations")

ax1.set_ylabel("Log Cost")
plt.title("Effect of Network Pruning on Cost Function")

'''
x_regular = list(range(len(nnet['Cost'])))
x_modified = list(range(iterations))
print(x_modified)
ax1.plot(x_regular,z)
ax1_ticks = list(range(0,(iterations+2)*4000,4000))
print(len(z))
print(ax1_ticks)
#ax1.set_xticks(ax1_ticks)
ax2.set_xlim(ax1.get_xlim())
'''

'''
fig = plt.plot(np.log(nnet['Cost']))
plt.xlabel("Iterations")


for xv in xvline:
    plt.axvline(xv,linestyle='--', dashes = (5,180), color='black')
'''



plt.show()
