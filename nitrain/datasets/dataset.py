import os
import warnings
import numpy as np
import math
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

        self._inputs = inputs
        self._outputs = outputs
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

    def split(self, p, shuffle=False):
        n_vals = len(self)
        indices = np.arange(n_vals)
        train_indices = indices[:math.ceil(n_vals*p)]
        test_indices = indices[math.ceil(n_vals*p):]
        
        ds_train = Dataset(self._inputs,
                           self._outputs,
                           self.transforms,
                           self._base_dir,
                           self._base_file)

        ds_test = Dataset(self._inputs,
                          self._outputs,
                          self.transforms,
                          self._base_dir,
                          self._base_file)
        
        #print(f'Before. {len(ds_test.inputs.values)}')
        ds_train.inputs = ds_train.inputs.select(train_indices)
        ds_train.outputs = ds_train.outputs.select(train_indices)
        #
        #
        #print(f'After. {len(ds_test.inputs)}')
        ds_test.inputs = ds_test.inputs.select(test_indices)
        ds_test.outputs = ds_test.outputs.select(test_indices)
        
        return ds_train, ds_test
        
    def filter(self, expr):
        raise NotImplementedError('Not implemented')
    
    def prefetch(self):
        raise NotImplementedError('Not implemented')
    
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


