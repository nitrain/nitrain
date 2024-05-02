import glob
import os
from parse import parse
from fnmatch import fnmatch

import pandas as pd
import numpy as np

# list of ntimage images
class ImageReader:
    def __init__(self, images, label=None):
        """
        Examples
        --------
        >>> import ntimage as nt
        >>> from nitrain.readers import ImageReader
        >>> img = nt.load(nt.example_data('r16'))
        >>> imgs = [img, img, img]
        >>> reader = ImageReader(imgs)
        >>> value = reader[1]
        """
        self.values = images
        self.ids = None
        self.label = label

    def select(self, idx):
        self.values = [self.values[i] for i in idx]
        
    def map_values(self, base_dir=None, base_file=None, base_label=None):
        if self.label is None:
            if base_label is not None:
                self.label = base_label
            else:
                self.label = 'image'

    def __getitem__(self, idx):
        return {self.label: self.values[idx]}

    def __len__(self):
        return len(self.values)