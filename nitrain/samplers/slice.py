import numpy as np
import random
import math

import ntimage as nti

    
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
            if nti.is_image(self.y[0]):
                self.y = [self.y[i] for i in indices]
            else:
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
    
    def __repr__(self):
        return f'''SliceSampler(axis={self.axis}, batch_size={self.batch_size}, shuffle={self.shuffle})'''



def create_slices(inputs, outputs, axis):
    # TODO: let slice sampler be applied selectively via dictionary
    # right now, all images in the inputs / outputs will be sliced
    new_inputs = []
    new_outputs = []
    for tmp_input, tmp_output in zip(inputs, outputs):
        if nti.is_image(tmp_input):
            slices = tmp_input.shape[axis]
        else:
            if nti.is_image(tmp_input[0]):
                slices = tmp_input[0].shape[axis]
        
        for i in range(slices):
            if isinstance(tmp_input, list):
                new_inputs.append([x.slice(i, axis) if nti.is_image(x) else x for x in tmp_input])
            else:
                new_inputs.append(tmp_input.slice(i, axis) if nti.is_image(tmp_input) else tmp_input)
            
            if isinstance(tmp_output, list):
                new_outputs.append([x.slice(i, axis) if nti.is_image(x) else x for x in tmp_output])
            else:
                new_outputs.append(tmp_output.slice(i, axis) if nti.is_image(tmp_output) else tmp_output)
            
    return new_inputs, new_outputs