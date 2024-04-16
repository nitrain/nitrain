import warnings
import os
import glob
import pandas as pd
import numpy as np
from fnmatch import fnmatch
from google.cloud import storage
from google.oauth2 import service_account
from parse import parse
import ants

from .. import platform

class PlatformDataset:
    
    def __init__(self,
                 name,
                 x,
                 y,
                 transforms=None,
                 token=None):
        """
        Initialize a dataset stored on the nitrain.dev platform. 
        
        You cannot access dataset records from a PlatformDataset like you
        would any other dataset. This class is "lazy" in the sense that
        records are only read in locally once you pass the dataset to a 
        LocalTrainer or call the `.to_local()` method to download it. 
        
        The PlatformDataset is most useful if you are using the PlatformTrainer to
        train a model with the nitrain.dev platform. 
        
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
        >>> from nitrain.datasets import PlatformDataset
        >>> dataset = PlatformDataset(name='ds000711', 
                                      x={'pattern': '*/anat/*_T1w.nii.gz', 'exclude': '**run-02*'},
                                      y={'file': 'participants.tsv', 'column': 'age'})
        >>> dataset = PlatformDataset(name='ds000711', 
                                      x={'filenames': ['sub-001/anat/T1.nii.gz', '...']},
                                      y={'file': 'participants.tsv', 'column': 'age'})
        """
        # convert base_dir (nick-2/first-dataset) to gcs fuse
        if token is None:
            token = os.environ.get('NITRAIN_API_TOKEN')
            if token is None:
                raise Exception('No api token given or found. Set `NITRAIN_API_TOKEN` or create an account to get your token.')

        # this will raise exception if token is not valid
        user = platform._get_user_from_token(token)
        base_dir = f'{user}/{name}'
        
        # TODO: check if dataset record exists
        
        self.name = name
        self.x = x
        self.y = y
        self.transforms = transforms
        self.token = token

    def to_local(self, folder=None):
        """
        Download a PlatformDataset to local file storage.
        
        This function will download only the records specified with the
        x and y config. It will return the same dataset as a FolderDataset
        class that can be used locally.
        
        Arguments
        ---------
        folder : string
            If specified, the dataset will be downloaded to this folder. Otherwise,
            it will be downloaded to the nitrain home directory (~/.nitrain)
            
        Returns
        -------
        """
        raise NotImplementedError('Not implemented yet.')
        
    def __getitem__(self, idx):
        raise NotImplementedError('Not implemented yet.')
    
    def __len__(self):
        raise NotImplementedError('Not implemented yet.')
    
    def __copy__(self):
        raise NotImplementedError('Not implemented yet.')
    
    def __repr__(self):
        raise NotImplementedError('Not implemented yet.')