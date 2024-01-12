import Base as base
import matplotlib.pyplot as plt

nnet = base.load_nnet()

plt.plot(nnet['Cost'])
plt.show()