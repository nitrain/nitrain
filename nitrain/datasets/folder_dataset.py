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
        x_config = x
        y_config = y
        
        pattern = x_config['pattern']
        glob_pattern = pattern.replace('{id}','*')
        x = sorted(glob.glob(glob_pattern, root_dir=base_dir))
        if 'exclude' in x_config.keys():
            x = [file for file in x if not fnmatch(file, x_config['exclude'])]
        
        # TODO: support '{id}/*' but 
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
        self.datalad = datalad
        self.x_config = x_config
        self.y_config = y_config
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms
        self.participants = participants
        self.x = x
        self.x_ids = x_ids
        self.y = y
        self.y_ids = y_ids
        
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
        ds = dl.Dataset(path = self.base_dir)
        res = ds.get(files)
        
        for file in files:
            img = ants.image_read(file)
            
            if self.x_transforms:
                for x_tx in self.x_transforms:
                    img = x_tx(img)
            
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
            path=self.base_dir,
            x=self.x,
            y=self.y,
            x_transforms=self.x_transforms
        )
    
