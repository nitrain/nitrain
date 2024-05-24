import os
import warnings
import numpy as np
import math
import random
from copy import deepcopy, copy

from ..readers.utils import infer_reader
from .utils import reduce_to_list, apply_transforms

class Dataset:
    
    def __init__(self, inputs, outputs, transforms=None, base_dir=None, base_file=None):
        """
        Create a nitrain dataset from data in memory or on the local filesystem.
        
        Examples
        --------
        import nitrain as nt
        from nitrain import readers
        dataset = nt.Dataset(
            inputs = readers.ImageReader('~/desktop/ds004711/sub-*/anat/*_T1w.nii.gz'),
            outputs = readers.ColumnReader('~/desktop/ds004711/participants.tsv', 'age'),
        )
        """

        inputs = infer_reader(inputs)
        outputs = infer_reader(outputs)

        self._base_dir = base_dir
        self._base_file = base_file
        
        if base_dir:
            base_dir = os.path.expanduser(base_dir)
        
        inputs.map_values(base_dir=base_dir, base_file=base_file, base_label='inputs')
        outputs.map_values(base_dir=base_dir, base_file=base_file, base_label='outputs')
        
        # ensure alignment
        if len(inputs.values) != len(outputs.values):
            warnings.warn('Inputs and outputs do not have same length. This could be misalignment between mappings.')

        self.inputs = inputs
        self.outputs = outputs
        self.transforms = transforms

    def select(self, n, random=False):
        """
        Select a number of records from the dataset.
        """
        all_indices = np.arange(len(self))
        if random:
            selected_indices = np.random.choice(all_indices, size=n, replace=False)
        else:
            selected_indices = np.arange(n)
            
        ds = deepcopy(self)
        ds.inputs = ds.inputs.select(selected_indices)
        ds.outputs = ds.outputs.select(selected_indices)
        
        return ds
        
    def split(self, p, random=False):
        """
        Split dataset into training, testing, and optionally validation.
        
        dataset.split(0.8)
        dataset.split((0.8,0.2))
        dataset.split((0.8,0.1,0.1))
        """
        if isinstance(p, float):
            p = (p, 1-p, 0)
        
        if isinstance(p, (list, tuple)):
            if len(p) == 2:
                p = p + (0,)
        
        if sum(p) != 1:
            raise Exception('The probabilities must sum to 1.')
            
        n_vals = len(self)
        indices = np.arange(n_vals)
        
        if random:
            if p[2] > 0:
                sampled_indices = np.random.choice([0,1,2], size=n_vals, p=p)
                train_indices = np.where(sampled_indices==0)[0]
                test_indices = np.where(sampled_indices==1)[0]
                val_indices = np.where(sampled_indices==2)[0]
            else:
                sampled_indices = np.random.choice([0,1], size=n_vals, p=p[:-1])
                train_indices = np.where(sampled_indices==0)[0]
                test_indices = np.where(sampled_indices==1)[0]
        else:
            if p[2] > 0:
                train_indices = indices[:math.ceil(n_vals*p[0])]
                test_indices = indices[math.ceil(n_vals*p[0]):math.ceil(n_vals*(p[0]+p[1]))]
                val_indices = indices[math.ceil(n_vals*(p[0]+p[1])):]
            else:
                train_indices = indices[:math.ceil(n_vals*p[0])]
                test_indices = indices[math.ceil(n_vals*p[0]):]
            
        ds_train = deepcopy(self)
        ds_test = deepcopy(self)
            
        ds_train.inputs = ds_train.inputs.select(train_indices)
        ds_train.outputs = ds_train.outputs.select(train_indices)
        ds_test.inputs = ds_test.inputs.select(test_indices)
        ds_test.outputs = ds_test.outputs.select(test_indices)

        if p[2] > 0:
            ds_val = deepcopy(self)
            ds_val.inputs = ds_val.inputs.select(val_indices)
            ds_val.outputs = ds_val.outputs.select(val_indices)
            return ds_train, ds_test, ds_val
        else:
            return ds_train, ds_test
    
    def __getitem__(self, idx):
        reduce = True
        if isinstance(idx, tuple):
            reduce = idx[1]
            idx = idx[0]
            
        if isinstance(idx, slice):
            idx = list(range(idx.stop)[idx])
            is_slice = True
        else:
            idx = [idx]
            is_slice = False
            
        x_items = []
        y_items = []
        for i in idx:
            x_raw = self.inputs[i]
            y_raw = self.outputs[i]
            
            if self.transforms:
                for tx_name, tx_value in self.transforms.items():
                    x_raw, y_raw = apply_transforms(tx_name, tx_value, x_raw, y_raw)
            
            if reduce:
                x_raw = reduce_to_list(x_raw)
                y_raw = reduce_to_list(y_raw)
            
            x_items.append(x_raw)
            y_items.append(y_raw)
        
        if not is_slice:
            x_items = x_items[0]
            y_items = y_items[0]

        return x_items, y_items
    
    def __len__(self):
        return len(self.inputs)
    
    def __repr__(self):
        s = 'Dataset (n={})\n'.format(len(self))
        
        s = s +\
            '     {:<10} : {}\n'.format('Inputs', self.inputs)+\
            '     {:<10} : {}\n'.format('Outputs', self.outputs)+\
            '     {:<10} : {}\n'.format('Transforms', len(self.transforms) if self.transforms else '{}')
        return s


