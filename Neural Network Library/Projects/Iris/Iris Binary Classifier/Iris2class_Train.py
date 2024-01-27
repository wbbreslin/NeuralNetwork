from Projects.Iris import Iris2class as data
import TrainingAlgorithms as train
import copy
import Base as base

nnet = copy.deepcopy(data.nnet)
nnet = train.gradient_descent(nnet, step_size=0.05, max_iterations=1000)
base.store_nnet(nnet)