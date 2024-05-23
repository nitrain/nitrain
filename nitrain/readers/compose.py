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
        
    def select(self, idx):
        new_reader = ComposeReader(self.readers)
        new_reader.values = self.values
        new_reader.readers = [reader.select(idx) for reader in new_reader.readers]
        new_reader.values = list(zip(*[reader.values for reader in new_reader.readers]))
        return new_reader
        
    def map_gcs_values(self, base_dir=None, base_file=None, base_label=None, bucket=None, credentials=None):
        for idx, reader in enumerate(self.readers):
            reader.map_gcs_values(base_dir=base_dir, base_file=base_file, base_label=f'{base_label}-{idx}',
                                  bucket=bucket, credentials=credentials)
        
        if self.label is None:
            if base_label is not None:
                self.label = base_label
            else:
                self.label = 'compose'
        self.values = list(zip(*[reader.values for reader in self.readers]))
    
    def map_values(self, base_dir=None, base_file=None, base_label=None):
        for idx, reader in enumerate(self.readers):
            reader.map_values(base_dir=base_dir, base_file=base_file, base_label=f'{base_label}-{idx}')
        
        if self.label is None:
            if base_label is not None:
                self.label = base_label
            else:
                self.label = 'compose'
        self.values = list(zip(*[reader.values for reader in self.readers]))

    def __getitem__(self, idx):
        values = {}
        for reader in self.readers:
            values.update(reader[idx])
        return {self.label: values}
    
    def __len__(self):
        return len(self.values)

