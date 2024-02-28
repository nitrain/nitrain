# datasets help you map the location of your images

# Datasets determine WHERE the data is stored:
# - locally in files
# - non-locally in S3 or Github
# - in memory via numpy arrays

# Improving performance:
# https://www.tensorflow.org/guide/data_performance

import copy
import os
import json
import ants
import bids
import nibabel
import datalad.api as dl
import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from .. import utils


class CSVDataset:
    
    def __init__(self,
                 path, 
                 x_config,
                 y_config,
                 x_transform=None):
        
        data = pd.read_csv(path)
        x = list(data[x_config['column']].to_numpy())
        y = data[y_config['column']].to_numpy()
        
        self.path = path
        self.x_config = x_config
        self.y_config = y_config
        self.x_transform = x_transform
        self.data = data
        self.x = x
        self.y = y

    def filter(self, expr):
        raise NotImplementedError('Not implemented yet')

    def precompute_transforms(self, expr):
        raise NotImplementedError('Not implemented yet')
    
    def __getitem__(self, idx):
        files = self.x[idx]
        if not isinstance(idx, slice):
            files = [files]
            
        y = self.y[idx]
        
        x = []
        for file in files:
            img = ants.image_read(file)
        
            if self.x_transform is not None:
                img = self.x_transform(img)
            
            x.append(img)
            
        
        if not isinstance(idx, slice):
            x = x[0]

        return x, y
    
    def __len__(self):
        return len(self.x)
    
    def __copy__(self):
        return CSVDataset(
            path=self.path,
            x_config=self.x_config,
            y_config=self.y_config,
            x_transform=self.x_transform
        )
    