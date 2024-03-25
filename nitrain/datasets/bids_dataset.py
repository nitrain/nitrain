import warnings
import copy
import os
import json
import ants
import bids
import datalad.api as dl
import numpy as np
import pandas as pd
import sys

from .. import utils

class BIDSDataset:
    
    def __init__(self,
                 base_dir, 
                 x,
                 y,
                 x_transforms=None,
                 y_transforms=None,
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
        >>> dataset = FolderDataset('ds000711', 
                                    x={'datatype': 'anat', 'suffix': 'T1w'},
                                    y={'file':'participants.tsv', 'column':'age'})
        >>> model = nitrain.models.fetch_pretrained('t1-brainage', finetune=True)
        >>> model.fit(dataset)
        """
        
        if layout is None:
            if 'scope' in x.keys():
                layout = bids.BIDSLayout(base_dir, derivatives=True)
            else:
                layout = bids.BIDSLayout(base_dir, derivatives=False)
        
        x_config = x
        y_config = y
        
        # GET X
        ids = layout.get(return_type='id', target='subject', **x_config)
        x = layout.get(return_type='filename', **x_config)
        if len(x) == 0:
            raise Exception('No images found matching the specified x.')
            
            
        participants_file = layout.get(suffix='participants', extension='tsv')[0]
        participants = pd.read_csv(participants_file, sep='\t')
        p_col = participants.columns[0] # assume participant id is first row
        p_suffix = 'sub-' # assume participant col starts with 'sub-'
        participants = participants[participants[p_col].isin([p_suffix+id for id in ids])]
        y = participants[y_config['column']].to_numpy()

        if len(x) != len(y):
            warnings.warn(f'len(x) [{len(x)}] != len(y) [{len(y)}]. Do some participants have multiple runs?')
        
        self.base_dir = base_dir
        self.x_config = x_config
        self.y_config = y_config
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms
        self.layout = layout
        self.participants = participants
        self.x = x
        self.y = y
        self.datalad = datalad

    def __getitem__(self, idx):
        files = self.x[idx]
        if not isinstance(idx, slice):
            files = [files]
        y = self.y[idx]
        
        if self.y_transforms is not None:
            for y_tx in self.y_transforms:
                y = y_tx(y)
        
        # make sure files are downloaded
        if self.datalad:
            ds = dl.Dataset(path = self.base_dir)
            res = ds.get(files)
        
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
    
    def __repr__(self):
        pass
    
    def __copy__(self):
        return BIDSDataset(
            base_dir=self.base_dir,
            x=self.x,
            y=self.y,
            x_transforms=self.x_transforms,
            y_transforms=self.y_transforms,
            datalad=self.datalad,
            layout=self.layout
        )
    
