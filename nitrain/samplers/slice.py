import numpy as np
import random
import math

import ants

    
class SliceSampler:
    """
    Sampler that returns batches of 2D slices from 3D images.
    """
    def __init__(self, batch_size=24, axis=-1, shuffle=False):
        self.batch_size = batch_size
        self.axis = axis
        self.shuffle = shuffle
    
    def __call__(self, x, y):
        # create slices of all images
        self.x = create_slices(x, self.axis)
        self.y = create_slices(y, self.axis)
        
        xx = self.x[0]
        if isinstance(xx, list):
            while isinstance(xx, list):
                batch_length = len(xx)
                xx = xx[0]
        else:
            batch_length = len(self.x)

        self.batch_length = batch_length
        self.n_batches = math.ceil(batch_length / self.batch_size)
                
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
    
    def __repr__(self):
        return f'''SliceSampler(axis={self.axis}, batch_size={self.batch_size}, shuffle={self.shuffle})'''

def select_items(x, idx):
    if isinstance(x[0], list):
        return [select_items(xx, idx) for xx in x]
    else:
        return x[idx]

def create_slices(x, axis):
    def flatten_extend(matrix):
        flat_list = []
        for row in matrix:
            flat_list.extend(row)
        return flat_list
    
    if isinstance(x[0], list):
        return [create_slices([x[i][j] for i in range(len(x))], axis) for j in range(len(x[0]))]
    if ants.is_image(x[0]):
        return flatten_extend([[xx.slice(i, axis) for i in range(xx.shape[axis])] for xx in x])
    else:
        return x
