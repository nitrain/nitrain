import math
import numpy as np
import torch
import ants
import types

from ..datasets.random_transform_dataset import RandomTransformDataset

class DatasetLoader:
    
    def __init__(self, 
                 dataset, 
                 batch_size, 
                 x_transforms=None, 
                 y_transforms=None, 
                 co_transforms=None,
                 shuffle=False):
        """
        Arguments
        ---------
        
        Examples
        --------
        ds = Dataset()
        ld = DatasetLoader(ds)
        xb, yb = next(iter(ld))

        """
        #transform_dataset = RandomTransformDataset(dataset, x_transforms=x_transforms)
        #self._dataset = transform_dataset
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
       batch_size = self.batch_size
       dataset = self.dataset
       n_batches = math.ceil(len(dataset) / batch_size)
       
       # perform random transforms
       
       # gather slices
       
       # shuffle images / slices
       
       batch_idx = 0
       while batch_idx < n_batches:
           data_indices = slice(batch_idx*batch_size, min((batch_idx+1)*batch_size, len(dataset)))
           x, y = dataset[data_indices]
           yield np.array([xx.numpy() for xx in x]), y
           batch_idx += 1

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
    
