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


class PlatformDataset:
    
    def __init__(self,
                 base_dir,
                 x,
                 y,
                 x_transforms=None,
                 y_transforms=None,
                 fuse=False,
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
        # convert base_dir (nick-2/first-dataset) to gcs fuse
        if fuse:
            base_dir = os.path.join('/gcs/ants-dev/datasets/', base_dir)
        else:
            base_dir = os.path.join('datasets/', base_dir)
            
        x_config = x
        y_config = y
        
        pattern = x_config['pattern']
        glob_pattern = pattern.replace('{id}','*')
        x = sorted(glob.glob(os.path.join(glob_pattern, base_dir)))
        x = [os.path.relpath(xx, base_dir) for xx in x]
        if 'exclude' in x_config.keys():
            x = [file for file in x if not fnmatch(file, x_config['exclude'])]
        
        if '{id}' in pattern:
            x_ids = [parse(pattern.replace('*','{other}'), file).named['id'] for file in x]
        else:
            x_ids = [xx.split('/')[0] for xx in x]
        x = [os.path.join(base_dir, file) for file in x]
        
        # GET Y
        participants_file = os.path.join(base_dir, y_config['file'])
        participants = pd.read_csv(participants_file, sep='\t')
        
        # match x and y ids
        p_col = participants.columns[0] # assume participant id is first row
        participants = participants.sort_values(p_col)
        all_y_ids = participants[p_col].to_numpy()
        if len(x_ids) != len(all_y_ids):
            warnings.warn(f'Mismatch between x ids {len(x_ids)} and y ids {len(all_y_ids)} - finding intersection')
        y_ids = sorted(list(set(x_ids) & set(all_y_ids)))
        
        participants = participants[participants[p_col].isin(y_ids)]
        y = participants[y_config['column']].to_numpy()
        
        # remove x values that are not found in y
        x = [x[idx] for idx in range(len(x)) if x_ids[idx] in y_ids]
        x_ids = y_ids


        if len(x) != len(y):
            warnings.warn(f'len(x) [{len(x)}] != len(y) [{len(y)}]. Do some participants have multiple runs?')
        
        self.base_dir = base_dir
        self.fuse = fuse
        self.x_config = x_config
        self.y_config = y_config
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms
        self.participants = participants
        self.x = x
        self.x_ids = x_ids
        self.y = y
        self.y_ids = y_ids
        
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
        return f'''PlatformDataset(name = "{self.name}",
                    x = {self.x_config},
                    y = {self.y_config},
                    x_transforms = [{tx_repr}],
                    y_transforms = {self.y_transforms},
                    fuse = {self.fuse},
                    credentials = None)'''