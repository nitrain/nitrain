import numpy as np
import random
import math

from .patch import create_patches
from .slice import create_slices

class SlicePatchSampler:
    """
    Sampler that returns 2D patches from 3D images.
    """
    def __init__(self, patch_size, stride, axis, batch_size, shuffle=False):
        
        if isinstance(patch_size, int):
            patch_size = [patch_size, patch_size]
        
        if isinstance(stride, int):
            stride = [stride, stride]
            
        self.patch_size = patch_size
        self.stride = stride
        self.axis = axis
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __call__(self, x, y):
        # create slices of all images
        x = create_slices(x, self.axis)
        y = create_slices(y, self.axis)
        # then create patches from all those slices
        x, y = create_patches(x, y, self.patch_size, self.stride)
        
        self.x = x
        self.y = y
        self.n_batches = math.ceil(len(x) / self.batch_size)
                
        return self

    def __iter__(self):
        """
        Apply shuffling whenever the sampler is instantiated
        as an iterator.
        """
        self.idx = 0
        
        # apply shuffling
        if self.shuffle:
            indices = random.sample(range(len(self.y)), len(self.y))
            x = self.x
            self.x = [x[i] for i in indices]
            self.y = self.y[indices]
            
        return self

    def __next__(self):
        if self.idx < self.n_batches:
            data_indices = slice(self.idx*self.batch_size, min((self.idx+1)*self.batch_size, len(self.x)))
            self.idx += 1
            x = self.x[data_indices]
            y = self.y[data_indices]
            return x, y
        else:
            raise StopIteration
