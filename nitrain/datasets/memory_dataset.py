import copy
import os
import json
import ants
import bids
import nibabel
import datalad.api as dl
import numpy as np
import pandas as pd


from .. import utils


class MemoryDataset:
    
    def __init__(self, x, y, x_transforms=None, y_transforms=None, co_transforms=None):
        self.x = x
        self.y = y
        
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
        
    def filter(self, expr):
        raise NotImplementedError('Not implemented yet')

    def precompute(self):
        raise NotImplementedError('Not implemented yet')
    
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

    def __len__(self):
        return len(self.x)
    
    def __str__(self):
        return f'MemoryDataset with {len(self.x)} records'

    def __repr__(self):
        return f'<MemoryDataset with {len(self.x)} records>'
    
