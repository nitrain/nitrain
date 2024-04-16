import glob
import os
from parse import parse
from fnmatch import fnmatch

import datalad.api as dl
import pandas as pd
import numpy as np
import ants

class ComposeReader:
    def __init__(self, readers):
        self.readers = readers
        values = [reader.values for reader in self.readers]
        self.values = list(zip(*values))
        
        # TODO: align ids for composed readers
        if self.readers[0].ids is not None:
            self.ids = self.readers[0].ids
        else:
            self.ids = None

    def __getitem__(self, idx):
        return [reader[idx] for reader in self.readers]

