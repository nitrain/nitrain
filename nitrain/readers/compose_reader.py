import glob
import os
from parse import parse
from fnmatch import fnmatch

import pandas as pd
import numpy as np

class ComposeReader:
    def __init__(self, readers, label=None):
        self.readers = readers
        self.label = label
    
    def map_values(self, base_dir=None, base_label=None):
        for idx, reader in enumerate(self.readers):
            reader.map_values(base_dir=base_dir, base_label=f'{base_label}-{idx}')
        
        if self.label is None:
            if base_label is not None:
                self.label = base_label
            else:
                self.label = 'compose'

    def __getitem__(self, idx):
        values = {}
        for reader in self.readers:
            values.update(reader[idx])
        return {self.label: values}

