import warnings
import os
import glob
import tempfile
import pandas as pd
import numpy as np
from fnmatch import fnmatch
from google.cloud import storage
from google.oauth2 import service_account
from torch.utils.data import Dataset
import textwrap

import ants

from .. import utils


class PlatformDataset:
    
    def __init__(self,
                 name,
                 x,
                 y,
                 x_transforms=None,
                 y_transforms=None,
                 credentials=None):
        """
        Initialize a nitrain dataset consisting of local filepaths.
        
        Arguments
        ---------
        x : dict or list of dicts
            The config dict to specify how input files should be found. This generally
            takes one of two forms:
            - pattern-based: {'pattern': '*/anat/*_T1w.nii.gz'}
            - filename-based: {'filenames': ['sub-001/anat/sub-001_T1w.nii.gz', '...']}
            If the input to the model is multiple images, then x should be a list of configs:
            e.g., [{'pattern': '*/anat/*_T1w.nii.gz'}, {'pattern': '*/anat/*_T2w.nii.gz'}] 
            
        Example
        -------
        >>> dataset = PlatformDataset(name='ds000711', 
                                      x={'pattern': '*/anat/*_T1w.nii.gz', 'exclude': '**run-02*'},
                                      y={'file': 'participants.tsv', 'column': 'age'})
        >>> dataset = PlatformDataset(name='ds000711', 
                                      x={'filenames': ['sub-001/anat/T1.nii.gz', '...']},
                                      y={'file': 'participants.tsv', 'column': 'age'})
        """
        self.name = name
        self.x_config = x
        self.y_config = y
        self.x = None
        self.y = None
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms
        
    def initialize(self):
        pass

    def __getitem__(self, idx):
        files = self.x[idx]
        if not isinstance(idx, slice):
            files = [files]
        y = self.y[idx]
        
        if self.y_transforms is not None:
            y = np.array([self.y_transforms(yy) for yy in y])
        
        x = []
        for file in files:
            # if on vertex, file will be available to access. otherwise, download it to local tmp dir.
            if True:
                local_filepath = file
            else:
                local_filepath = os.path.join(self.tmp_dir.name, file)
                if not os.path.exists(local_filepath):
                    os.makedirs('/'.join(local_filepath.split('/')[:-1]), exist_ok=True)
                    file_blob = self.bucket_client.blob(file)
                    file_blob.download_to_filename(local_filepath)
            
            img = ants.image_read(local_filepath)
        
            if self.x_transforms:
                for x_tx in self.x_transforms:
                    img = x_tx(img)
            
            x.append(img)
        
        if not isinstance(idx, slice):
            x = x[0]

        return x, y
    
    def __len__(self):
        return len(self.x)
    
    def __copy__(self):
        return PlatformDataset(
            name = self.name,
            x = self.x_config,
            y = self.y_config,
            x_transforms = self.x_transforms,
            y_transforms = self.y_transforms,
            credentials = self.credentials
        )
    
    def __repr__(self):
        tx_repr = ', '.join([repr(x_tx) for x_tx in self.x_transforms])
        return f'''datasets.PlatformDataset(name = "{self.name}",
                    x = {self.x_config},
                    y = {self.y_config},
                    x_transforms = [{tx_repr}],
                    y_transforms = {self.y_transforms},
                    credentials = None)'''