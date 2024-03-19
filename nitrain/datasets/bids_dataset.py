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

    def filter(self, expr, inplace=False):
        """
        Filter the dataset by column values in the participants file
        """
        ds = copy.copy(self)
        
        participants = ds.participants.query(expr)
        
        p_col = participants.columns[0] # assume participant id is first row
        p_suffix = 'sub-' # assume participant col starts with 'sub-'
        query_ids = [id.split('-')[1] for id in participants[p_col]]
        
        file_ids = ds.layout.get(return_type='id', target='subject', **ds.x)
        ids = sorted(list(set(file_ids).intersection(query_ids)))

        # only keep ids that are in the participants file
        x = ds.layout.get(return_type='filename', subject=ids, **ds.x)
        
        # GET Y
        p_col = participants.columns[0] # assume participant id is first row
        p_suffix = 'sub-' # assume participant col starts with 'sub-'
        participants = participants[participants[p_col].isin([p_suffix+id for id in ids])]
        y = participants[ds.y['column']].to_numpy()
        
        # make changes to instance
        ds.participants = participants
        ds.x = x
        ds.y = y
        return ds
    
    def precompute_transforms(self, desc='precompute'):
        """
        Compute all transforms on the input images and save
        them to file as a derivative. The original filepaths
        will be replaced with paths to the transformed images.
        Precomputing transforms is a good idea if you plan to
        read from disk.
        """
        if self.x_transforms is None:
            raise Exception('No transforms set, so nothing to precompute.')
        
        # create derivatives/nitrain directory if necessary
        derivatives_dir = os.path.join(self.base_dir, 'derivatives/')
        save_dir = os.path.join(derivatives_dir, 'nitrain/')

        if not os.path.exists(derivatives_dir):
            os.mkdir(derivatives_dir)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            
            data_description = {"Name": "nitrain precomputed transforms",
                                "BIDSVersion": "v1.8.0 (2022-10-29)", 
                                "DatasetType": "derivatives",
                                "GeneratedBy": [{"Name": "nitrain precomputed transforms"}]}
            with open(os.path.join(save_dir, "dataset_description.json"), "w") as outfile: 
                json.dump(data_description, outfile, indent=1)
        
        # make sure files are downloaded
        files = self.x
        with utils.SilentFunction():
            ds = dl.Dataset(path = self.base_dir)
            res = ds.get(files)
        
        for file in files:
            img = ants.image_read(file)
            img = self.x_transforms(img)
            
            file_ending = file.replace(f'{self.base_dir}/', '')
            
            try:
                os.makedirs(os.path.dirname(os.path.join(save_dir, file_ending)))
            except:
                pass
            
            # add `desc` entity manually because `layout.build_path` doesnt work with new entities?
            suffix = self.layout.parse_file_entities(file)['suffix']
            save_filename = os.path.join(save_dir, 
                                         file_ending.replace(suffix, 
                                                             f'desc-{desc}_{suffix}'))
            ants.image_write(img, save_filename)
        
        # add derivative layout
        try:
            self.layout.add_derivatives(os.path.join(self.base_dir, 'derivatives/nitrain'))
        except:
            pass
        
        # replace existing filename with the transformed ones from derivatives/nitrain
        config = self.x_config
        config['desc'] = desc
        self.x = self.layout.derivatives['nitrain'].get(return_type='filename', **config)
        self.x = config
        self.x_transforms = None        

    def __getitem__(self, idx):
        files = self.x[idx]
        y = self.y[idx]
        if not isinstance(idx, slice):
            files = [files]
            y = np.array([y])
        
        if self.y_transforms is not None:
            for y_tx in self.y_transforms:
                y = np.array([y_tx(yy) for yy in y])
        
        # make sure files are downloaded
        #if self.datalad:
        #    with SilentFunction():
        #        ds = dl.Dataset(path = self.base_dir)
        #        res = ds.get(files)
        
        x = []
        for file in files:
            img = ants.image_read(file)
        
            if self.x_transforms:
                for x_tx in self.x_transforms:
                    img = x_tx(img)
            
            x.append(img)
        
        if not isinstance(idx, slice):
            x = x[0]
            y = y[0]

        return x, y
    
    def __len__(self):
        return len(self.x)
    
    def __repr__(self):
        pass
    
    def __copy__(self):
        return BIDSDataset(
            path=self.base_dir,
            x=self.x,
            y=self.y,
            x_transforms=self.x_transforms,
            layout=self.layout
        )
    
