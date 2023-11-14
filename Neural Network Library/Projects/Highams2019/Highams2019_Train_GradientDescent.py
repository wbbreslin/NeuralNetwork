import Highams2019 as data
import TrainingAlgorithms as train
import copy

nnet = copy.deepcopy(data.nnet)
nnet = train.gradient_descent(nnet, max_iterations=5000)