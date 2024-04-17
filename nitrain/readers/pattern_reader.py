import glob
import os
from parse import parse
from fnmatch import fnmatch

import pandas as pd
import numpy as np
import ntimage as nt

class PatternReader:
    def __init__(self, pattern, base_dir=None, exclude=None):
        """
        >>> import ntimage as nt
        >>> from nitrain.readers import PatternReader
        >>> reader = PatternReader('~/Desktop/kaggle-liver-ct/volumes/*.nii')
        >>> img = reader[1]
        """
        pattern = os.path.expanduser(pattern)
        glob_pattern = pattern.replace('{id}','*')
        
        if base_dir is not None:
            if not base_dir.endswith('/'):
                base_dir += '/'
            base_dir = os.path.expanduser(base_dir)
            glob_pattern = os.path.join(base_dir, glob_pattern)
        x = sorted(glob.glob(glob_pattern, recursive=True))

        if base_dir is not None:
            x = [os.path.relpath(xx, base_dir) for xx in x]
        
        if exclude:
            x = [file for file in x if not fnmatch(file, exclude)]

        if '{id}' in pattern:
            ids = [parse(pattern.replace('*','{other}'), file).named['id'] for file in x]
        else:
            ids = None

        if base_dir is not None:
            x = [os.path.join(base_dir, file) for file in x]
        
        if len(x) == 0:
            raise Exception(f'No filepaths found that match {glob_pattern}')

        self.base_dir = base_dir
        self.pattern = glob_pattern
        self.exclude = exclude
        self.values = x
        self.ids = ids
        
    def __getitem__(self, idx):
        filename = self.values[idx]
            
        return nt.load(self.values[idx])
    
