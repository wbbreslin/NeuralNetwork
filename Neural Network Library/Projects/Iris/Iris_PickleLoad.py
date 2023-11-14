import Base as base

nnet = base.load_nnet()
print(nnet.keys())
print(nnet['States'][-1])