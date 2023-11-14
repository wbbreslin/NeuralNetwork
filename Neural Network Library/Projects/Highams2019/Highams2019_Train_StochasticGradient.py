import Highams2019 as data
import TrainingAlgorithms as train
import copy

nnet = copy.deepcopy(data.nnet)
nnet = train.stochastic_gradient(nnet, subset_size=5, max_iterations=10000)