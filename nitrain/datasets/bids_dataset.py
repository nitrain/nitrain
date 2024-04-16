import warnings
import copy
import os
import re
import json
import ants
import bids
import datalad.api as dl
import numpy as np
import pandas as pd
import sys

from .readers import infer_reader

class BIDSDataset:
    
    def __init__(self,
                 base_dir, 
                 x,
                 y,
                 transforms=None,
                 datalad=False,
                 layout=None):
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
        >>> dataset = BIDSDataset('ds000711', 
                                    x={'datatype': 'anat', 'suffix': 'T1w'},
                                    y={'file':'participants.tsv', 'column':'age'})
        """
        
        if layout is None:
            if 'scope' in x.keys():
                layout = bids.BIDSLayout(base_dir, derivatives=True)
            else:
                layout = bids.BIDSLayout(base_dir, derivatives=False)
        
        x_reader = infer_reader(x, base_dir)
        y_reader = infer_reader(y, base_dir)
        
        self.x_reader = x_reader
        self.y_config = y_reader
        self.x = x_reader.values
        self.y = y_reader.values
        
        self.base_dir = base_dir
        self.transforms = transforms
        self.layout = layout
        self.datalad = datalad
    
    def __len__(self):
        return len(self.x)
    
    def __str__(self):
        return f'FolderDataset with {len(self.x)} records'

    def __repr__(self):
        # TODO: implement transforms __repr__
        if self.transforms:
            tx_str = 'transforms=transforms'
            
        if self.transforms is not None:
            text = f"""FolderDataset(base_dir = '{self.base_dir}',
                    x = {self._x_arg},
                    y = {self._y_arg},
                    {tx_str})"""
        else:
            text = f"""FolderDataset(base_dir = '{self.base_dir}',
                    x = {self._x_arg},
                    y = {self._y_arg})"""   
        
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub('[\n]+', '\n', text)
        return text
    
    def __copy__(self):
        return BIDSDataset(
            base_dir=self.base_dir,
            x=self.x,
            y=self.y,
            transforms=self.transforms,
            datalad=self.datalad,
            layout=self.layout
        )
    
