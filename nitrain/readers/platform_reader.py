import glob
import os
from parse import parse
from fnmatch import fnmatch

import datalad.api as dl
import pandas as pd
import numpy as np
import ants

class PlatformReader:
    def __init__(self, bucket, base_dir, pattern, exclude=None, fuse=False, lazy=False):
        pass
    def __getitem__(self):
        pass