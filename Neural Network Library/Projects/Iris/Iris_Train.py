import Iris as data
import TrainingAlgorithms as train
import copy
import Base as base

nnet = copy.deepcopy(data.nnet)
nnet = train.gradient_descent(nnet, step_size=0.1, max_iterations=10**2)
base.store_nnet(nnet)