import numpy as np
import random
import math

def create_slices(images, values, axis):
    slices = []
    new_values = []
    for image, value in zip(images, values):
        for i in range(image.shape[axis]):
            slices.append(image.slice_image(axis, i))
            new_values.append(value)
            
    return slices, np.array(new_values)
    
    
class SliceSampler:
    """
    Standard sampler that just returns the batch with or without shuffling
    
    Examples
    --------
    
    """
    def __init__(self, batch_size, axis=0, shuffle=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.axis = axis
        self.x = None
        self.y = None
    
    def __call__(self, x, y):
        # create slices of all images
        self.x, self.y = create_slices(x, y, self.axis)
        self.n_batches = math.ceil(len(self.x) / self.batch_size)
                
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
    
    def __len__(self):
        """
        number of batches from the sampler
        """
        return 1