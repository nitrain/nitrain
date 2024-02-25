# Datasets determine WHERE the data is stored:
# - locally in files
# - non-locally in S3 or Github
# - in memory via numpy arrays

import os
import bids
import nibabel
import datalad.api as dl
import numpy as np
import pandas as pd

from .. import utils
# layout.get_subjects()
# layout.get_datatypes()
# layout.get(datatype='anat', suffix='T1w',  return_type='filename')
        
class S3Dataset:
    pass

class GithubDataset:
    pass

class FileDataset:
    
    def __init__(self,
                 path, 
                 layout,
                 X_config,
                 y_config):
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
        >>> dataset = FileDataset('ds000711', X_datatype='anat', X_suffix='T1w', y_column='age')
        >>> model = nitrain.models.fetch_pretrained('t1-brainage', finetune=True)
        >>> model.fit(dataset)
        """
        
        self.path = path
        
        if layout == 'bids':
            self.layout = bids.BIDSLayout(path)
        else:
            self.layout = layout    
        self.X_config = X_config
        self.y_config = y_config
    
    def fetch_data(self, n=None):
        layout = self.layout
        files = layout.get(return_type='filename',
                           **self.X_config)
        if n is not None:
            files = files[:n]
        
        # make sure files are downloaded
        ds = dl.Dataset(path = self.path)
        res = ds.get(files)
        
        X = utils.files_to_array(files)
        
        # handle y
        if self.y_config.get('filename'):
            df = pd.read_csv(os.path.join(ds.path, self.y_config['filename']), sep='\t')
            y = df[self.y_config['column']].to_numpy()
            
            if n is not None:
                y = y[:n]
        
        return X, y

"""
# download relevant T1 images
layout = bids.BIDSLayout(ds.path)
files = layout.get(extension='nii.gz', datatype='anat', suffix='T1w', return_type='filename')[:50]
res = ds.get(files)

# read participants file
df = pd.read_csv(os.path.join(ds.path, 'participants.tsv'), sep='\t')
df = df[df['participant_id'].isin(['sub-'+s for s in layout.get_subjects()[:50]])]

# get final arrays
X = utils.files_to_array(files)
y = df['age'].to_numpy() # (50,)
"""


class Dataset:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y