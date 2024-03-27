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


class FolderDataset:
    
    def __init__(self,
                 base_dir, 
                 x,
                 y,
                 x_transforms=None,
                 y_transforms=None,
                 datalad=False):
        """
        Initialize a nitrain dataset consisting of local filepaths.
        
        Arguments
        ---------
        x : dict or list of dicts
        y : dict or list of dicts
        x_transforms : transform or list of transform
        y_transforms : transform or list of transform
        datalad : boolean

        
        Example
        -------
        >>> dataset = FolderDataset('ds000711', 
                                    x={'pattern': '**_T1w.nii.gz'},
                                    y={'file':'participants.tsv', 'column':'age'})
        >>> model = nitrain.models.fetch_pretrained('t1-brainage', finetune=True)
        >>> model.fit(dataset)
        """
        if base_dir.startswith('~'):
            base_dir = os.path.expanduser(base_dir)
            
        x_config = x
        y_config = y
        
        
        # GET X
        x, x_ids = _get_filepaths_from_pattern_config(x_config, base_dir)
        
        # GET Y
        if 'pattern' in y_config.keys():
            # images
            y, y_ids = _get_filepaths_from_pattern_config(y_config, base_dir)
            participants = None
        elif 'file' in y_config.keys():
            participants_file = os.path.join(base_dir, y_config['file'])
            if participants_file.endswith('.tsv'):
                participants = pd.read_csv(participants_file, sep='\t')
            elif participants_file.endswith('.csv'):
                participants = pd.read_csv(participants_file)
                
            # TODO: match x and y ids
            y = participants[y_config['column']].to_numpy()
        else:
            raise Exception('The y argument should have a pattern or file key.')
        
        # remove x values that are not found in y
        if x_ids is not None and y_ids is not None:
            x = [x[idx] for idx in range(len(x)) if x_ids[idx] in y_ids]
        
        ids = x_ids

        if len(x) != len(y):
            warnings.warn(f'len(x) [{len(x)}] != len(y) [{len(y)}]. Do some participants have multiple runs?')
        
        self.base_dir = base_dir
        self.datalad = datalad
        self.x_config = x_config
        self.y_config = y_config
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms
        
        self.participants = participants
        
        self.x = x
        self.y = y
        self.ids = ids

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
    
    def __copy__(self):
        return FolderDataset(
            base_dir=self.base_dir,
            x=self.x_config,
            y=self.y_config,
            x_transforms=self.x_transforms,
            y_transforms=self.y_transforms,
            datalad=self.datalad
        )
    

def _get_filepaths_from_pattern_config(config, base_dir):
    pattern = config['pattern']
    glob_pattern = pattern.replace('{id}','*')
    x = sorted(glob.glob(os.path.join(base_dir, glob_pattern)))
    x = [os.path.relpath(xx, base_dir) for xx in x]
    if 'exclude' in config.keys():
        x = [file for file in x if not fnmatch(file, config['exclude'])]

    if '{id}' in pattern:
        x_ids = [parse(pattern.replace('*','{other}'), file).named['id'] for file in x]
    else:
        x_ids = None
        
    x = [os.path.join(base_dir, file) for file in x]
    
    return x, x_ids