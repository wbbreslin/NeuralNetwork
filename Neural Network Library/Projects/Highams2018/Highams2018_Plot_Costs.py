import matplotlib.pyplot as plt
import Highams2018 as highams

plt.plot(highams.nnet.costs)
plt.xlabel("Training Iterations")
plt.ylabel("Cost")
plt.title("Network Training via Gradient Descent")
plt.show()