import glob
import os
from parse import parse
from fnmatch import fnmatch

import pandas as pd
import numpy as np
import ntimage as nt


# numpy array that must be converted to images
class ArrayReader:
    def __init__(self, array, label=None):
        """
        Examples
        --------
        >>> import numpy as np
        >>> from nitrain.readers import ArrayReader
        >>> arr = np.random.normal(40,10,(10,50,50,50))
        >>> reader = ArrayReader(arr)
        >>> img = reader[1]
        """
        self.array = array
        self.label = label
    
    def map_values(self, base_dir=None, base_label=None):
        array = self.array
        if array.ndim > 2:
            # arrays must be converted to images
            array_list = np.split(array, array.shape[0])
            ns = array.shape[1:]
            self.values = [nt.from_numpy(tmp.reshape(*ns)) for tmp in array_list]
        else:
            self.values = array
        
        if self.label is None:
            if base_label is not None:
                self.label = base_label
            else:
                self.label = 'array'
        
    def __getitem__(self, idx):
        return {self.label: self.values[idx]}

    def __len__(self):
        return len(self.values)