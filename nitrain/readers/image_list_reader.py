import glob
import os
from parse import parse
from fnmatch import fnmatch

import pandas as pd
import numpy as np

# list of ntimage images
class ImageListReader:
    def __init__(self, images):
        """
        Examples
        --------
        >>> import ntimage as nt
        >>> from nitrain.readers import ImageListReader
        >>> img = nt.load(nt.example_data('r16'))
        >>> imgs = [img, img, img]
        >>> reader = ImageListReader(imgs)
        >>> value = reader[1]
        """
        self.values = images
        self.ids = None

    def __getitem__(self, idx):
        return self.values[idx]
