import numpy as np
import random
import math


class BlockSampler:
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
        self.x, self.y = create_patches(x, y, self.block_size, self.stride)
        self.n_batches = math.ceil(len(self.x) / self.batch_size)
                
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

def create_patches(images, values, block_size, stride):
    cropped_images = []
    new_values = []
    for image, value in zip(images, values):
        # extract all blocks
        x_strides = np.arange(0, (image.shape[0]-block_size[0]+1), step=stride[0])
        y_strides = np.arange(0, (image.shape[1]-block_size[1]+1), step=stride[1])
        z_strides = np.arange(0, (image.shape[2]-block_size[2]+1), step=stride[2])
        
        grid = np.meshgrid(x_strides, y_strides, z_strides)
        x_indices = grid[0].flatten()
        y_indices = grid[1].flatten()
        z_indices = grid[2].flatten()
        
        for a, b, c in zip(x_indices, y_indices, z_indices):
            cropped_image = image.crop_indices((a,b,c),
                                               (a+block_size[0],
                                                b+block_size[1],
                                                c+block_size[2]))
            cropped_images.append(cropped_image)
            new_values.append(value)
                
    return cropped_images, np.array(new_values)