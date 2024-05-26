import numpy as np
import random
import math

import ants

class BaseSampler:
    """
    Standard sampler that just returns the batch with or without shuffling
    
    Examples
    --------
    
    """
    def __init__(self, batch_size, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __call__(self, x, y):
        self.x = rearrange_values(x)
        self.y = rearrange_values(y)

        xx = self.x[0]
        if isinstance(xx, list):
            while isinstance(xx, list):
                batch_length = len(xx)
                xx = xx[0]
        else:
            batch_length = len(self.x)
        
        self.n_batches = math.ceil(len(self.x) / self.batch_size)
        self.batch_length = batch_length
        
        return self

    def __iter__(self):
        """
        Get a sampled batch
        """
        self.idx = 0
        
        # apply shuffling
        if self.shuffle:
            indices = random.sample(range(len(self.y)), len(self.y))
            self.x = [self.x[i] for i in indices]
            self.y = [self.y[i] for i in indices]
            
        return self

    def __next__(self):
        if self.idx < self.n_batches:
            data_indices = slice(self.idx*self.batch_size, min((self.idx+1)*self.batch_size, self.batch_length))
            self.idx += 1
            x = select_items(self.x, data_indices)
            y = select_items(self.y, data_indices)
            return x, y
        else:
            raise StopIteration

def select_items(x, idx):
    if isinstance(x[0], list):
        return [select_items(xx, idx) for xx in x]
    else:
        return x[idx]
        
def rearrange_values(x):
    if isinstance(x[0], list):
        return [rearrange_values([x[i][j] for i in range(len(x))]) for j in range(len(x[0]))]
    return x