import glob
import os
from parse import parse
from fnmatch import fnmatch

import datalad.api as dl
import pandas as pd
import numpy as np
import ntimage as nt


# numpy array that must be converted to images
class ArrayReader:
    def __init__(self, array):
        """
        x = np.random.normal(40,10,(10,50,50,50))
        x_config = ArrayReader(x)
        """
        self.array = array
        
        if array.ndim > 2:
            # arrays must be converted to images
            array_list = np.split(array, array.shape[0])
            ns = array.shape[1:]
            self.values = [ants.from_numpy(tmp.reshape(*ns)) for tmp in array_list]
        else:
            self.values = array
        
    def __getitem__(self, idx):
        return self.values[idx]
