import glob
import os
from parse import parse
from fnmatch import fnmatch

import datalad.api as dl
import pandas as pd
import numpy as np

# list of ants images
class ImageReader:
    def __init__(self, images):
        self.values = images
        self.ids = None

    def __getitem__(self, idx):
        return self.values[idx]
