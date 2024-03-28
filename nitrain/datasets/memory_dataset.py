import copy
import os
import json
import ants
import bids
import nibabel
import datalad.api as dl
import numpy as np
import pandas as pd

from .base_dataset import BaseDataset
from .configs import _infer_config

class MemoryDataset(BaseDataset):
    
    def __init__(self, x, y, x_transforms=None, y_transforms=None, co_transforms=None):
        """
        Examples
        --------
        import numpy as np
        from nitrain.datasets import MemoryDataset
        dataset = MemoryDataset(
            np.random.normal(20,10,(5,50,50)),
            np.random.normal(20,10,5)
        )
        x, y = dataset[0]
        """
        
        x_config = _infer_config(x)
        y_config = _infer_config(y)
        
        if x_transforms is not None:
            if not isinstance(x_transforms, list):
                x_transforms = [x_transforms]

        if y_transforms is not None:
            if not isinstance(y_transforms, list):
                y_transforms = [y_transforms]

        if co_transforms is not None:
            if not isinstance(co_transforms, list):
                co_transforms = [co_transforms]          

        self.x_transforms = x_transforms
        self.y_transforms = y_transforms
        self.co_transforms = co_transforms

        self.x_config = x_config
        self.y_config = y_config
        self.x = x_config.values
        self.y = y_config.values
    
    def __getitem__(self, idx):
        images = self.x[idx]
        if not isinstance(idx, slice):
            images = [images]
        y = self.y[idx]
        
        if self.y_transforms is not None:
            for y_tx in self.y_transforms:
                y = y_tx(y)
        
        x = []
        for image in images:
        
            if self.x_transforms:
                for x_tx in self.x_transforms:
                    image = x_tx(image)
            
            x.append(image)
        
        if not isinstance(idx, slice):
            x = x[0]

        return x, y
    
    def __str__(self):
        return f'MemoryDataset with {len(self.x)} records'

    def __repr__(self):
        return f'<MemoryDataset with {len(self.x)} records>'
    
