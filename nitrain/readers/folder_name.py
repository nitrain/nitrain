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

class FolderNameReader:
    """
    Returns the name of the folder for files based on the included pattern.
    
    This is useful if your images are stored in folders whose names have meaning.
    For instance:
    
    cat/
        img1.jpeg
        img2.jpeg
        ...
    dog/
        img1.jpeg
        img2.jpeg
        ...
    
    Creating a `FolderNameReader('*/*.jpeg')` would return 'cat' or 'dog' for the 
    appropriate image. This pairs well with an ImageReader:
    
    dataset = nt.Dataset(ImageReader('*/*.jpeg'), FolderNameReader('*/*.jpeg'))
    x, y = dataset[0] # x is img1 loaded in memory; y is 'cat'
    """
    def __init__(self, pattern, base_dir=None, exclude=None, label=None, level=0, format='string'):
        """
        >>> import ants
        >>> from nitrain.readers import ImageReader
        >>> reader = FolderNameReader('volumes/*.nii')
        >>> reader.map_values(base_dir='~/Desktop/kaggle-liver-ct/')
        >>> img = reader[1]
        """
        self.pattern = os.path.expanduser(pattern)
        
        if base_dir:
            base_dir = os.path.expanduser(base_dir)
        self.base_dir = base_dir
        self.exclude = exclude
        self.label = label
        self.level = level
        self.format = format
    
    def select(self, idx):
        new_reader = FolderNameReader(self.pattern, self.base_dir, self.exclude, self.label, self.level)
        new_reader.values = self.values
        new_reader.values = [new_reader.values[i] for i in idx]
        return new_reader
        
    def map_gcs_values(self, bucket, credentials=None, base_dir=None, base_file=None, base_label=None):
        if base_dir is None:
            base_dir = self.base_dir
        
        pattern = self.pattern
        exclude = self.exclude
        level = self.level
        
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
        
        if len(x) == 0:
            raise Exception(f'No filepaths found that match {glob_pattern}')

        values = [xx.split('/')[level] for xx in x]
        unique_values = np.unique(values)
        
        if self.format == 'integer':
            self.values = [np.where(unique_values==v)[0][0] for v in values]
        elif self.format == 'onehot':
            self.values = [list(np.eye(len(unique_values),
                                       dtype='uint32')[np.where(unique_values==v)[0][0]]) for v in values]
        elif self.format == 'string':
            self.values = values
        else:
            raise Exception('The format value must be `integer`, `onehot`, or `string`.')
        
        self.unique_values = list(unique_values)
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
        level = self.level
        
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
        
        if len(x) == 0:
            raise Exception(f'No filepaths found that match {glob_pattern}')
        
        values = [xx.split('/')[level] for xx in x]
        unique_values = np.unique(values)
        
        if self.format == 'integer':
            self.values = [np.where(unique_values==v)[0][0] for v in values]
        elif self.format == 'onehot':
            self.values = [list(np.eye(len(unique_values),
                                       dtype='uint32')[np.where(unique_values==v)[0][0]]) for v in values]
        elif self.format == 'string':
            self.values = values
        else:
            raise Exception('The format value must be `integer`, `onehot`, or `string`.')
        
        self.unique_values = list(unique_values)
        self.ids = ids
        
        if self.label is None:
            if base_label is not None:
                self.label = base_label
            else:
                self.label = 'folder_name'
                
    def __getitem__(self, idx):
        return {self.label: np.array(self.values[idx])}
    
    def __len__(self):
        return len(self.values)