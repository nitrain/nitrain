import warnings
import os
import tempfile
import pandas as pd
import numpy as np
from fnmatch import fnmatch
from google.cloud import storage
from google.oauth2 import service_account
from torch.utils.data import Dataset

import ants

from .. import utils


class GoogleCloudDataset:
    
    def __init__(self,
                 bucket, 
                 base_dir,
                 x,
                 y,
                 x_transforms=None,
                 y_transforms=None,
                 credentials=None,
                 use_fuse=False):
        """
        Initialize a nitrain dataset consisting of local filepaths.
        
        Arguments
        ---------
        
        Example
        -------
        >>> dataset = GoogleCloudDataset(bucket='ants-dev',
                                         base_dir='datasets/nick_2/ds000711', 
                                         x={'pattern': '*/anat/*_T1w.nii.gz', 'exclude': '**run-02*'},
                                         y={'file': 'participants.tsv', 'column': 'age'},
                                         credentials='/Users/ni5875cu/Desktop/ants.dev/engine/deep-dynamics-415608-4046316ec2f1.json')
        """
        if isinstance(credentials, str):
            credentials = service_account.Credentials.from_service_account_file(credentials)
        if credentials is not None:
            storage_client = storage.Client(credentials=credentials)
        bucket_client = storage_client.bucket(bucket)

        x_config = x
        y_config = y
        
        # GET X
        if not base_dir.endswith('/'): base_dir += '/'
        x_blobs = storage_client.list_blobs(bucket, match_glob=f'{base_dir}{x_config["pattern"]}')
        x = list([blob.name.replace(base_dir, '') for blob in x_blobs])
        if 'exclude' in x_config.keys():
            x = [file for file in x if not fnmatch(file, x_config['exclude'])]
        x_ids = [xx.split('/')[0] for xx in x]
        x = [os.path.join(base_dir, file) for file in sorted(x)]
        
        if len(x) == 0:
            raise Exception('Did not find any x values corresponding to the x config.')

        # GET Y
        y_file = os.path.join(base_dir, y_config['file'])
        y_blob = bucket_client.blob(y_file)
        tmp_file = tempfile.NamedTemporaryFile()
        y_blob.download_to_filename(tmp_file.name)
        y_df = pd.read_csv(tmp_file.name, sep='\t')
        tmp_file.close()

        # match x and y ids
        p_col = y_df.columns[0] # assume participant id is first row
        y_df = y_df.sort_values(p_col)
        all_y_ids = y_df[p_col].to_numpy()
        if len(x_ids) != len(all_y_ids):
            warnings.warn(f'Mismatch between x ids {len(x_ids)} and y ids {len(all_y_ids)} - finding intersection')
        y_ids = sorted(list(set(x_ids) & set(all_y_ids)))

        y_df = y_df[y_df[p_col].isin(y_ids)]
        y = y_df[y_config['column']].to_numpy()

        # remove x values that are not found in y
        x = [x[idx] for idx in range(len(x)) if x_ids[idx] in y_ids]
        x_ids = y_ids
        
        self.credentials = credentials
        self.storage_client = storage_client
        self.bucket_client = bucket_client
        self.bucket = bucket
        self.base_dir = base_dir
        self.x_config = x_config
        self.y_config = y_config
        self.x = x
        self.y = y
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.use_fuse = use_fuse

    def __getitem__(self, idx):
        files = self.x[idx]
        if not isinstance(idx, slice):
            print('not a slice')
            files = [files]
        y = self.y[idx]
        
        if self.y_transforms is not None:
            y = np.array([self.y_transforms(yy) for yy in y])
        
        x = []
        for file in files:
            # if on vertex, file will be available to access. otherwise, download it to local tmp dir.
            if self.use_fuse:
                local_filepath = os.path.join('/gcs/', self.bucket, file)
                print(local_filepath)
            else:
                local_filepath = os.path.join(self.tmp_dir.name, file)
                if not os.path.exists(local_filepath):
                    os.makedirs('/'.join(local_filepath.split('/')[:-1]), exist_ok=True)
                    file_blob = self.bucket_client.blob(file)
                    file_blob.download_to_filename(local_filepath)
            
            img = ants.image_read(local_filepath)
        
            if self.x_transforms is not None:
                img = self.x_transforms(img)
            
            x.append(img)
        
        if not isinstance(idx, slice):
            x = x[0]

        return x, y
    
    def __len__(self):
        return len(self.x)
    
    def __copy__(self):
        return GoogleCloudDataset(
            bucket = self.bucket,
            base_dir = self.base_dir,
            x = self.x_config,
            y = self.y_config,
            x_transforms = self.x_transforms,
            y_transforms = self.y_transforms,
            credentials = self.credentials
        )
    
