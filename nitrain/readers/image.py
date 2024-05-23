import glob
import os
from parse import parse
from fnmatch import fnmatch
import glob
from google.cloud import storage
from google.oauth2 import service_account

import pandas as pd
import numpy as np
import ants

class ImageReader:
    def __init__(self, pattern, base_dir=None, exclude=None, label=None):
        """
        >>> import ants
        >>> from nitrain.readers import ImageReader
        >>> reader = ImageReader('volumes/*.nii')
        >>> reader.map_values(base_dir='~/Desktop/kaggle-liver-ct/')
        >>> img = reader[1]
        """
        self.pattern = os.path.expanduser(pattern)
        
        if base_dir:
            base_dir = os.path.expanduser(base_dir)
        self.base_dir = base_dir
        self.exclude = exclude
        self.label = label
    
    def select(self, idx):
        new_reader = ImageReader(self.pattern, self.base_dir, self.exclude, self.label)
        new_reader.values = self.values
        new_reader.values = [new_reader.values[i] for i in idx]
        return new_reader
        
    def map_gcs_values(self, bucket, credentials=None, base_dir=None, base_file=None, base_label=None):
        if base_dir is None:
            base_dir = self.base_dir
        
        pattern = self.pattern
        exclude = self.exclude
        
        glob_pattern = pattern.replace('{id}','*')
        
        if base_dir is not None:
            if not base_dir.endswith('/'):
                base_dir += '/'
            glob_pattern = os.path.join(base_dir, glob_pattern)
        
        # GCS
        if isinstance(credentials, str):
            credentials = service_account.Credentials.from_service_account_file(credentials)
        storage_client = storage.Client(credentials=credentials)
        bucket_client = storage_client.bucket(bucket)
        
        x = storage_client.list_blobs(bucket, match_glob=glob_pattern)
        
        x = list([blob.name.replace(base_dir, '') for blob in x])

        if exclude:
            x = [file for file in x if not fnmatch(file, exclude)]

        if '{id}' in pattern:
            ids = [parse(pattern.replace('*','{other}'), file).named['id'] for file in x]
        else:
            ids = None

        if base_dir is not None:
            x = [os.path.join(base_dir, file) for file in x]
        
        if len(x) == 0:
            raise Exception(f'No filepaths found that match {glob_pattern}')

        self.values = x
        self.ids = ids
        
        if self.label is None:
            if base_label is not None:
                self.label = base_label
            else:
                self.label = 'pattern'
                
    def map_values(self, base_dir=None, base_label=None, **kwargs):
        if base_dir is None:
            base_dir = self.base_dir
        
        pattern = self.pattern
        exclude = self.exclude
        
        glob_pattern = pattern.replace('{id}','*')
        
        if base_dir is not None:
            if not base_dir.endswith('/'):
                base_dir += '/'
            glob_pattern = os.path.join(base_dir, glob_pattern)

        x = sorted(glob.glob(glob_pattern, recursive=True))
        
        if base_dir is not None:
            x = [os.path.relpath(xx, base_dir) for xx in x]
        
        if exclude:
            x = [file for file in x if not fnmatch(file, exclude)]

        if '{id}' in pattern:
            ids = [parse(pattern.replace('*','{other}'), file).named['id'] for file in x]
        else:
            ids = None

        if base_dir is not None:
            x = [os.path.join(base_dir, file) for file in x]
        
        if len(x) == 0:
            raise Exception(f'No filepaths found that match {glob_pattern}')

        self.values = x
        self.ids = ids
        
        if self.label is None:
            if base_label is not None:
                self.label = base_label
            else:
                self.label = 'pattern'
                
    def __getitem__(self, idx):
        return {self.label: ants.image_read(self.values[idx])}
    
    def __len__(self):
        return len(self.values)