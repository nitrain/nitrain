import glob
import os
from parse import parse
from fnmatch import fnmatch

import datalad.api as dl
import pandas as pd
import numpy as np
import ants

# one image from file
class PatternReader:
    def __init__(self, base_dir, pattern, exclude=None, datalad=False):
        if not base_dir.endswith('/'):
            base_dir += '/'
            
        glob_pattern = pattern.replace('{id}','*')
        glob_pattern = os.path.join(base_dir, glob_pattern)
        x = sorted(glob.glob(glob_pattern, recursive=True))
        x = [os.path.relpath(xx, base_dir) for xx in x]
        
        if exclude:
            x = [file for file in x if not fnmatch(file, exclude)]

        if '{id}' in pattern:
            ids = [parse(pattern.replace('*','{other}'), file).named['id'] for file in x]
        else:
            ids = None
            
        x = [os.path.join(base_dir, file) for file in x]
        
        if len(x) == 0:
            raise Exception(f'No filepaths found that match {glob_pattern}')

        self.base_dir = base_dir
        self.pattern = glob_pattern
        self.exclude = exclude
        self.datalad = datalad
        self.values = x
        self.ids = ids
        
    def __getitem__(self, idx):
        filename = self.values[idx]
        
        if self.datalad:
            ds = dl.Dataset(path = self.base_dir)
            res = ds.get(filename)
            
        return ants.image_read(self.values[idx])
    
