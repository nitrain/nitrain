import numpy as np
import random

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
        return self

    def __iter__(self):
        """
        Get a sampled batch
        """
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < 1:
            self.idx += 1
            x = self.x
            y = self.y            
            return x, y
        else:
            raise StopIteration
        
def rearrange_values(x):
    if isinstance(x[0], list):
        return [rearrange_values([x[i][j] for i in range(len(x))]) for j in range(len(x[0]))]
    return x