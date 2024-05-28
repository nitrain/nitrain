import numpy as np
import ants
import math

from .base import BaseSampler

class BlockSampler(BaseSampler):
    """
    Sampler that returns 3D blocks from 3D images.
    """
    def __init__(self, block_size, stride, batch_size, shuffle=False):
        
        if isinstance(block_size, int):
            block_size = [block_size, block_size, block_size]
        
        if isinstance(stride, int):
            stride = [stride, stride, stride]
            
        self.block_size = block_size
        self.stride = stride
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __call__(self, x, y):
        # create patches of all images
        self.x, self.y = create_blocks(x, y, self.block_size, self.stride)
        
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


def create_blocks(images, values, block_size, stride):
    new_inputs = []
    new_outputs = []
    for tmp_input, tmp_output in zip(images, values):
        # extract all blocks
        x_strides = np.arange(0, (tmp_input.shape[0]-block_size[0]+1), step=stride[0])
        y_strides = np.arange(0, (tmp_input.shape[1]-block_size[1]+1), step=stride[1])
        z_strides = np.arange(0, (tmp_input.shape[2]-block_size[2]+1), step=stride[2])
        
        grid = np.meshgrid(x_strides, y_strides, z_strides)
        x_indices = grid[0].flatten()
        y_indices = grid[1].flatten()
        z_indices = grid[2].flatten()
        
        for a, b, c in zip(x_indices, y_indices, z_indices):
            cropped_input = tmp_input.crop_indices((a,b,c),
                                               (a+block_size[0],
                                                b+block_size[1],
                                                c+block_size[2]))
            new_inputs.append(cropped_input)
            
            if ants.is_image(tmp_output):
                cropped_output = tmp_output.crop_indices((a,b,c),
                                                        (a+block_size[0],
                                                         b+block_size[1],
                                                         c+block_size[2]))
            else:
                cropped_output = tmp_output
            new_outputs.append(cropped_output)

    if not ants.is_image(tmp_output):
        new_outputs = np.array(new_outputs)
        
    return new_inputs, new_outputs