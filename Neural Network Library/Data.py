import Base as base
import numpy as np

class data:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def test_train_split(self,train_percent=0.8, seed=None):
        if train_percent < 1:
            rows = self.x.shape[0]
            indices = base.generate_random_indices(rows, random_seed=seed)
            split = int(np.round(rows * train_percent))
            train_indices = indices[0:split]
            test_indices = indices[split:]
            training_x = self.x[train_indices]
            training_y = self.y[train_indices]
            validation_x = self.x[test_indices]
            validation_y = self.y[test_indices]
            training = data(training_x, training_y)
            validation = data(validation_x, validation_y)
            return training, validation
        else:
            training = data(self.x, self.y)
            validation = []
            return training,validation
