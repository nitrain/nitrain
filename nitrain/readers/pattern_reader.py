import glob
import os
from parse import parse
from fnmatch import fnmatch

import pandas as pd
import numpy as np
import ntimage as nt

class PatternReader:
    def __init__(self, pattern, exclude=None, label=None):
        """
        >>> import ntimage as nt
        >>> from nitrain.readers import PatternReader
        >>> reader = PatternReader('volumes/*.nii')
        >>> reader.map_values(base_dir='~/Desktop/kaggle-liver-ct/')
        >>> img = reader[1]
        """
        self.pattern = pattern
        self.exclude = exclude
        self.label = label
    
    def map_values(self, base_dir=None, base_file=None, base_label=None):
        pattern = self.pattern
        exclude = self.exclude
        
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

        self.values = x
        self.ids = ids
        
        if self.label is None:
            if base_label is not None:
                self.label = base_label
            else:
                self.label = 'pattern'
                
    def __getitem__(self, idx):
        if not self.values:
            raise Exception('You must call `map_values()` before indexing a reader.')
        return {self.label: nt.load(self.values[idx])}
    
    def __len__(self):
        return len(self.values)