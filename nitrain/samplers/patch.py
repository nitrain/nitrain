import numpy as np
import ants
import math

from .base import BaseSampler

class PatchSampler(BaseSampler):
    """
    Sampler that returns strided patches from 2D images.
    """
    def __init__(self, patch_size, stride, batch_size, shuffle=False):
        
        if isinstance(patch_size, int):
            patch_size = [patch_size, patch_size]
        
        if isinstance(stride, int):
            stride = [stride, stride]
            
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __call__(self, x, y):
        # create patches of all images
        self.x, self.y = create_patches(x, y, self.patch_size, self.stride)
        
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


def create_patches(inputs, outputs, patch_size, stride):
    
    new_inputs = []
    new_outputs = []
    for tmp_input, tmp_output in zip(inputs, outputs):
        # extract all patches
        x_strides = np.arange(0, (tmp_input.shape[0]-patch_size[0]+1), step=stride[0])
        y_strides = np.arange(0, (tmp_input.shape[1]-patch_size[1]+1), step=stride[1])
        
        grid = np.meshgrid(x_strides, y_strides)
        x_indices = grid[0].flatten()
        y_indices = grid[1].flatten()
        
        for a, b in zip(x_indices, y_indices):
            cropped_input = tmp_input.crop_indices((a,b), (a+patch_size[0],b+patch_size[1]))
            new_inputs.append(cropped_input)
            
            if ants.is_image(tmp_output):
                cropped_output = tmp_output.crop_indices((a,b), (a+patch_size[0],b+patch_size[1]))    
            else:
                cropped_output = tmp_output
            new_outputs.append(cropped_output)

    if not ants.is_image(tmp_output):
        new_outputs = np.array(new_outputs)
        
    return new_inputs, new_outputs
