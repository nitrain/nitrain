import numpy as np
import random


class BaseSampler:
    """
    Standard sampler that just returns the batch with or without shuffling
    
    Examples
    --------
    
    """
    def __init__(self, shuffle=False):
        self.shuffle = shuffle
    
    def __call__(self, x, y):
        self.x = x
        self.y = y
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
                
            if self.shuffle:
                indices = random.sample(range(len(y)), len(y))
                x = [x[i] for i in indices]
                y = y[indices]
            
            return x, y
        else:
            raise StopIteration
    
    def __len__(self):
        """
        number of batches from the sampler
        """
        return 1