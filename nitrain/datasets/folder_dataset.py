import warnings
import os
import re

from .base_dataset import BaseDataset
from .configs import _infer_config, _align_configs

class FolderDataset(BaseDataset):
    
    def __init__(self,
                 base_dir, 
                 x,
                 y,
                 x_transforms=None,
                 y_transforms=None):
        """
        Initialize a nitrain dataset consisting of local filepaths.
        
        Arguments
        ---------
        base_dir : string
        x : dict or list of dicts
        y : dict or list of dicts
        x_transforms : transform or list of transforms
        y_transforms : transform or list of transforms

        Example
        -------
        >>> from nitrain.datasets import FolderDataset
        >>> dataset = FolderDataset('~/Desktop/openneuro/ds004711', 
                                    x=[{'pattern': '{id}/anat/*T1w.nii.gz', 
                                        'exclude': '**run-02*'},
                                        {'pattern': '{id}/anat/*T1w.nii.gz', 
                                        'exclude': '**run-02*'}],
                                    y={'file':'participants.tsv', 'column':'age', 
                                       'id': 'participant_id'})
        """
        if base_dir.startswith('~'):
            base_dir = os.path.expanduser(base_dir)
        
        x_config = _infer_config(x, base_dir)
        y_config = _infer_config(y, base_dir)
        
        if len(x_config.values) != len(y_config.values):
            warnings.warn(f'Found that len(x) [{len(x_config.values)}] != len(y) [{len(y_config.values)}]. Attempting to match ids.')
            x_config, y_config = _align_configs(x_config, y_config)
        
        self.base_dir = base_dir
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms

        self._x_arg = x
        self._y_arg = y
        self.x_config = x_config
        self.y_config = y_config
        self.x = x_config.values
        self.y = y_config.values

    def __str__(self):
        return f'FolderDataset with {len(self.x)} records'

    def __repr__(self):
        if self.x_transforms:
            tx_repr = '[' + ', '.join([repr(x_tx) for x_tx in self.x_transforms]) + ']'
            x_tx = f'x_transforms = {tx_repr},'
        else:
            x_tx = ''
        
        if self.y_transforms:
            tx_repr = '[' + ', '.join([repr(y_tx) for y_tx in self.y_transforms]) + ']'
            y_tx = f'y_transforms = {tx_repr},'
        else:
            y_tx = ''
        
        if self.x_transforms is not None or self.y_transforms is not None:
            text = f"""FolderDataset(base_dir = '{self.base_dir}',
                    x = {self._x_arg},
                    y = {self._y_arg},
                    {x_tx}
                    {y_tx})"""
        else:
            text = f"""FolderDataset(base_dir = '{self.base_dir}',
                    x = {self._x_arg},
                    y = {self._y_arg})"""   
        
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub('[\n]+', '\n', text) 
        return text
                    
    def __copy__(self):
        return FolderDataset(
            base_dir=self.base_dir,
            x=self.x_config,
            y=self.y_config,
            x_transforms=self.x_transforms,
            y_transforms=self.y_transforms
        )
    
