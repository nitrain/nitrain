# datasets help you map the location of your images

# Datasets determine WHERE the data is stored:
# - locally in files
# - non-locally in S3 or Github
# - in memory via numpy arrays

# Improving performance:
# https://www.tensorflow.org/guide/data_performance

import copy
import os
import ants
import bids
import nibabel
import datalad.api as dl
import numpy as np
import pandas as pd

from .. import utils

__all__ = [
    'S3Dataset',
    'GithubDataset',
    'FolderDataset',
    'MemoryDataset',
    'CSVDataset'
]

            
class MemoryDataset:
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def filter(self, query):
        pass
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.x.shape[0]


class FolderDataset:
    
    def __init__(self,
                 path, 
                 x_config,
                 y_config,
                 x_transform=None,
                 layout='bids'):
        """
        Initialize a nitrain dataset consisting of local filepaths.
        
        Arguments
        ---------
        X_datatype : string or n-tuple of strings
            the datatype which determines the input images for the model. if
            you supply a n-tupple, then it is assumed that you want to use
            n images as input to the model. See bids.BIDSLayout
            
        X_suffix : string or n-tuple of strings
            the suffix which determines the input images for the model. if
            you supply a n-tupple, then it is assumed that you want to use
            n images as input to the model. See bids.BIDSLayout
        
        Example
        -------
        >>> dataset = FolderDataset('ds000711', X_datatype='anat', X_suffix='T1w', y_column='age')
        >>> model = nitrain.models.fetch_pretrained('t1-brainage', finetune=True)
        >>> model.fit(dataset)
        """
        
        if isinstance(layout, str):
            if layout.lower() == 'bids':
                layout = bids.BIDSLayout(path)
            else:
                raise Exception('Only bids layouts are accepted right now.')
        
        # GET X
        ids = layout.get(return_type='id', target='subject', **x_config)
        x = layout.get(return_type='filename', **x_config)
        
        # GET Y
        participants_file = layout.get(suffix='participants', extension='tsv')[0]
        participants = pd.read_csv(participants_file, sep='\t')
        p_col = participants.columns[0] # assume participant id is first row
        p_suffix = 'sub-' # assume participant col starts with 'sub-'
        participants = participants[participants[p_col].isin([p_suffix+id for id in ids])]
        y = participants[y_config['column']].to_numpy()

        if len(x) != len(y):
            raise Exception(f'len(x) [{len(x)}] != len(y) [{len(y)}]. Do some participants have multiple runs?')
        
        self.path = path
        self.x_config = x_config
        self.y_config = y_config
        self.x_transform = x_transform
        self.layout = layout
        self.participants = participants
        self.x = x
        self.y = y
        
    def filter(self, expr, inplace=False):
        """
        Filter the dataset by column values in the participants file
        """
        ds = copy.copy(self)
        
        participants = ds.participants.query(expr)
        
        p_col = participants.columns[0] # assume participant id is first row
        p_suffix = 'sub-' # assume participant col starts with 'sub-'
        query_ids = [id.split('-')[1] for id in participants[p_col]]
        
        file_ids = ds.layout.get(return_type='id', target='subject', **ds.x_config)
        ids = sorted(list(set(file_ids).intersection(query_ids)))

        # only keep ids that are in the participants file
        x = ds.layout.get(return_type='filename', subject=ids, **ds.x_config)
        
        # GET Y
        p_col = participants.columns[0] # assume participant id is first row
        p_suffix = 'sub-' # assume participant col starts with 'sub-'
        participants = participants[participants[p_col].isin([p_suffix+id for id in ids])]
        y = participants[ds.y_config['column']].to_numpy()
        
        # make changes to instance
        ds.participants = participants
        ds.x = x
        ds.y = y
        return ds

    def __getitem__(self, idx):
        files = self.x[idx]
        if not isinstance(idx, slice):
            files = [files]
        y = self.y[idx]
        
        # make sure files are downloaded
        ds = dl.Dataset(path = self.path)
        res = ds.get(files)
        
        x = []
        for file in files:
            img = ants.image_read(file)
        
            if self.x_transform is not None:
                img = self.x_transform(img)
                if not isinstance(img, np.ndarray):
                    img = img.numpy()
            else:
                img = img.numpy()
            x.append(img)
        x = np.array(x, dtype='float32')
        
        if not isinstance(idx, slice):
            x = x[0]

        return x, y
    
    def __len__(self):
        return len(self.x)
    
    def __copy__(self):
        return FolderDataset(
            path=self.path,
            x_config=self.x_config,
            y_config=self.y_config,
            x_transform=self.x_transform,
            layout=self.layout
        )
    

class CSVDataset:
    pass

class S3Dataset:
    pass

class GithubDataset:
    pass