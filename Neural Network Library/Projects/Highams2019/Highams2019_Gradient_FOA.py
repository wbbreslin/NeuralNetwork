import Highams2019 as data
import TrainingAlgorithms as train

nnet = data.nnet
nnet = train.gradient_descent(nnet, max_iterations=4000)
print(nnet["Gradients"][0])