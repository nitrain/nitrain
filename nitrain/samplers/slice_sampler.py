import numpy as np
import random
import math

    
class SliceSampler:
    """
    Sampler that returns batches of 2D slices from 3D images.
    """
    def __init__(self, sub_batch_size, axis=0, shuffle=False):
        self.sub_batch_size = sub_batch_size
        self.axis = axis
        self.shuffle = shuffle
    
    def __call__(self, x, y):
        # create slices of all images
        self.x, self.y = create_slices(x, y, self.axis)
        self.n_batches = math.ceil(len(self.x) / self.sub_batch_size)
                
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
            data_indices = slice(self.idx*self.sub_batch_size, min((self.idx+1)*self.sub_batch_size, len(self.x)))
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
        if self.n_batches is not None:
            return self.n_batches
        else:
            return 0
    
    def __repr__(self):
        return f'''samplers.SliceSampler(axis={self.axis}, sub_batch_size={self.sub_batch_size}, shuffle={self.shuffle})'''


def create_slices(images, values, axis):
    slices = []
    new_values = []
    for image, value in zip(images, values):
        for i in range(image.shape[axis]):
            slices.append(image.slice_image(axis, i))
            new_values.append(value)
            
    return slices, np.array(new_values)