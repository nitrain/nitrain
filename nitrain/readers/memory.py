import glob
import os
from parse import parse
from fnmatch import fnmatch

import pandas as pd
import numpy as np
import ntimage as nti

class MemoryReader:
    def __init__(self, data, label=None):
        """
        Read from in-memory records.
        
        The records can be a numpy array, a list of images, a list of scalars, etc.
        """
        self.values = data
        self.label = label
        
        if nti.is_image(data[0]):
            self.as_image = True
        else:
            self.as_image = False
            
    def select(self, idx):
        self.values = [self.values[i] for i in idx]
        
    def map_values(self, base_dir=None, base_file=None, base_label=None):
        if self.label is None:
            if base_label is not None:
                self.label = base_label
            else:
                self.label = 'memory'
        
    def __getitem__(self, idx):
        return {self.label: self.values[idx]}

    def __len__(self):
        return len(self.values)