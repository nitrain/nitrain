import warnings
from parse import parse
import copy
import os
import json
import ants
import datalad.api as dl
import numpy as np
import pandas as pd
import glob
from fnmatch import fnmatch

from .. import utils


class CSVDataset:
    
    def __init__(self,
                 path, 
                 x,
                 y,
                 x_transforms=None,
                 y_transforms=None):
        """
        Initialize a nitrain dataset consisting of local filepaths.
        
        Arguments
        ---------
        x : dict or list of dicts
            Info used to grab the correct images from the folder. A list
            of dicts means you want to return multiple images. This is helpful
            if you need some other image(s) to help process the primary image - e.g.,
            you can supply a list of 2-dicts to read in T1w images + the associated
            mask. Then, you could use `x_transforms` to mask the T1w image and only
            return the masked T1w image from the dataset.

        
        Example
        -------
        >>> dataset = CSVDataset('~/desktop/test/participants.csv', 
                                 x={'images': 'filenames_2d'},
                                 y={'column': 'age'})
        """ 
        x_config = x
        y_config = y
        
        # get csv file
        participants_file = os.path.join(os.path.expanduser(path))
        if participants_file.endswith('.tsv'):
            participants = pd.read_csv(participants_file, sep='\t')
        elif participants_file.endswith('.csv'):
            participants = pd.read_csv(participants_file)

        x = participants[x_config['images']]
        y = participants[y_config['column']].to_numpy()

        if len(x) != len(y):
            warnings.warn(f'len(x) [{len(x)}] != len(y) [{len(y)}]. Do some participants have multiple runs?')
        
        self.path = path
        self.x_config = x_config
        self.y_config = y_config
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms
        self.participants = participants
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        files = self.x[idx]
        if not isinstance(idx, slice):
            files = [files]
        y = self.y[idx]
        
        if self.y_transforms is not None:
            for y_tx in self.y_transforms:
                y = y_tx(y)
        
        x = []
        for file in files:
            img = ants.image_read(file)
        
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
        return CSVDataset(
            path=self.path,
            x=self.x,
            y=self.y,
            x_transforms=self.x_transforms,
            y_transforms=self.y_transforms
        )