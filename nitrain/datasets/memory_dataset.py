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
    
    def __init__(self, x, y, x_transform=None):
        self.x = x
        self.y = y
        self.x_transform = x_transform
        
    def filter(self, expr):
        raise NotImplementedError('Not implemented yet')
    
    def __getitem__(self, idx):
        x = self.x[idx]
        if self.x_transform is not None:
            if isinstance(x, list):
                x = [self.x_transform(xx) for xx in x]
            else:
                x = self.x_transform(x)
        return x, self.y[idx]

    def __len__(self):
        return len(self.x)
    
    def __str__(self):
        return f'MemoryDataset with {len(self.x)} records'

    def __repr__(self):
        return f'<MemoryDataset with {len(self.x)} records>'
    
