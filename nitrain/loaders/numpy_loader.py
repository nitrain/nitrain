import math
import numpy as np
import torch
import ants
import types

from ..datasets.transform_dataset import TransformDataset

class NumpyLoader:
    
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
        slice_axis : integer (optional)
            If provided, the loader will serve 2D image slices from
            3D images along the slice_axis. One iteration of the loader
            will thus be n_images * n_slices samples instead of only n_images, so
            every slice of every image will be seen in one epoch. The
            batch size will also now be in terms of slices instead of images.
        """
        transform_dataset = TransformDataset(dataset, x_transforms=x_transforms)
        self._dataset = transform_dataset
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
       batch_size = self.batch_size
       dataset = self._dataset
       n_batches = math.ceil(len(dataset) / batch_size)
       
       batch_idx = 0
       while batch_idx < n_batches:
           data_indices = slice(batch_idx*batch_size, min((batch_idx+1)*batch_size, len(dataset)))
           x, y = dataset[data_indices]
           yield x, y
           batch_idx += 1

    def __len__(self):
        return len(self.dataset)
    
class NumpyLoaderOld:

    def __init__(self, dataset, batch_size, x_transforms=None, y_transforms=None, co_transforms=None, **kwargs):
        transform_dataset = TransformDataset(dataset, x_transforms=x_transforms)
        self._dataset = transform_dataset
        self.dataset = dataset
        self.batch_size = batch_size

    def __getitem__(self, idx):
        batch_size = self.batch_size
        dataset = self._dataset

        data_indices = slice(idx*batch_size, min((idx+1)*batch_size, len(dataset)))
        x, y = dataset[data_indices]
        return x, y
    
    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
