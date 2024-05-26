import numpy as np
import random
import math

import ants

from .base import BaseSampler
    
class SliceSampler(BaseSampler):
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

    def __repr__(self):
        return f'''SliceSampler(axis={self.axis}, batch_size={self.batch_size}, shuffle={self.shuffle})'''


def create_slices(x, axis):
    def flatten_extend(matrix):
        flat_list = []
        for row in matrix:
            flat_list.extend(row)
        return flat_list
    
    if isinstance(x[0], list):
        return [create_slices([x[i][j] for i in range(len(x))], axis) for j in range(len(x[0]))]
    if ants.is_image(x[0]):
        return flatten_extend([[xx.slice_image(axis, i) for i in range(xx.shape[axis])] for xx in x])
    else:
        return x
