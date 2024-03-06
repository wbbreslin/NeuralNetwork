import numpy as np
import Highams2018 as hig
import copy
from Data import data
import matplotlib.pyplot as plt
from Highams2018_FSO import sensitivity
from scipy.stats import spearmanr

OSEnet_ref = hig.nnet_OSE_ref
training_ref = hig.training
validation = hig.validation

OSE = []
hig.nnet.forward(validation)
hig.nnet.backward(validation)
unmodified_cost = hig.nnet.costs[-1]
print(unmodified_cost)

for i in range(10):
    OSEnet = copy.deepcopy(OSEnet_ref)
    x_OSE = np.delete(training_ref.x,i,axis=0)
    y_OSE = np.delete(training_ref.y,i,axis=0)
    df_OSE = data(x_OSE,y_OSE)
    OSEnet.train(df_OSE, max_iterations=10000, step_size=0.25)
    OSEval = copy.deepcopy(OSEnet)
    OSEval.forward(validation)
    OSEval.backward(validation)
    new_cost = OSEval.costs[-1]
    delta = new_cost - unmodified_cost
    print([new_cost, delta])
    OSE.append(delta)


OSE = np.array(OSE).reshape(-1,1)

#print(OSE)
#print(unmodified_cost)
#print(np.corrcoef(sensitivity.flatten(),OSE.flatten()))
#print(spearmanr(sensitivity.flatten(), OSE.flatten()))
plt.plot(OSE, label="OSE")
plt.plot(sensitivity, label="Sensitivity")
plt.legend()
plt.show()