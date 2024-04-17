import glob
import os
from parse import parse
from fnmatch import fnmatch

import datalad.api as dl
import pandas as pd
import numpy as np

class GoogleCloudReader:
    def __init__(self, bucket, base_dir, pattern, exclude=None, fuse=False, lazy=False):
        pass
    def __getitem__(self, idx):
        file = self.files[idx]
        
        if self.fuse:
            local_filepath = file
        else:
            local_filepath = os.path.join(self.tmp_dir.name, file)
            if not os.path.exists(local_filepath):
                os.makedirs('/'.join(local_filepath.split('/')[:-1]), exist_ok=True)
                file_blob = self.bucket_client.blob(file)
                file_blob.download_to_filename(local_filepath)
    