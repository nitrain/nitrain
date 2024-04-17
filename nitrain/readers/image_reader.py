import glob
import os
from parse import parse
from fnmatch import fnmatch

import pandas as pd
import numpy as np

# list of ntimage images
class ImageReader:
    def __init__(self, images):
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
    
    def map_values(self):
        pass

    def __getitem__(self, idx):
        return self.values[idx]
