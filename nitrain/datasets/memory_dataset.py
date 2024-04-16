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
from .readers import infer_reader

class MemoryDataset(BaseDataset):
    
    def __init__(self, x, y, transforms=None):
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
        
        x_reader = infer_reader(x)
        y_reader = infer_reader(y)

        self.x_reader = x_reader
        self.y_reader = y_reader
        self.x = x_reader.values
        self.y = y_reader.values
        
        self.transforms = transforms
    
    def __str__(self):
        return f'MemoryDataset with {len(self.x)} records'

    def __repr__(self):
        return f'<MemoryDataset with {len(self.x)} records>'
    
