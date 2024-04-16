import glob
import os
from parse import parse
from fnmatch import fnmatch

import datalad.api as dl
import pandas as pd
import numpy as np
import ants

class ColumnReader:
    def __init__(self, base_dir, file, column, id=None):
        filepath = os.path.join(base_dir, file)
        
        if not os.path.exists(filepath):
            raise Exception(f'No file found at {filepath}')
        
        if filepath.endswith('.tsv'):
            participants = pd.read_csv(filepath, sep='\t')
        elif filepath.endswith('.csv'):
            participants = pd.read_csv(filepath)
            
        values = participants[column].to_numpy()
        
        if id is not None:
            ids = list(participants[id].to_numpy())
        else:
            ids = None
        
        self.base_dir = base_dir
        self.values = values
        self.ids = ids
        self.file = filepath
        self.column = column

    def __getitem__(self, idx):
        return self.values[idx]
    