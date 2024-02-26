# loaders help you feed your data into deep learning models with transforms
import math
import numpy as np

class DatasetLoader:
    
    def __init__(self, dataset, batch_size, x_transforms=None, y_transforms=None, co_transforms=None):
        self.dataset = dataset
        self.batch_size = batch_size
    
    def __iter__(self):
       batch_idx = 0
       batch_size = self.batch_size
       dataset = self.dataset
       
       while batch_idx < self.n_batches():
           data_indices = slice(batch_idx*batch_size, min((batch_idx+1)*batch_size, len(dataset)))
           imgs, y = self.dataset[data_indices]
           x = np.array([img.numpy() for img in imgs], dtype='float32')
           x = np.expand_dims(x, -1)
           yield x, y
           batch_idx += 1
    
    def n_batches(self):
        return math.ceil(len(self.dataset) / self.batch_size)
    
    @property
    def input_shape(self):
        # load the first input, perform any transforms, and get shape
        return self.dataset[0][0].shape

class Counter:
    def __init__(self, start, end):
        self.num = start
        self.end = end
 
    def __iter__(self):
        return self
 
    def __next__(self): 
        if self.num > self.end:
            raise StopIteration
        else:
            self.num += 1
            return self.num - 1