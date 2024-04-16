import warnings
import os
import re

from .base_dataset import BaseDataset
from ..readers.utils import infer_reader, align_readers
from .. import platform

class FolderDataset(BaseDataset):
    
    def __init__(self,
                 base_dir, 
                 x,
                 y,
                 transforms=None):
        """
        Initialize a nitrain dataset consisting of local filepaths.
        
        Arguments
        ---------
        base_dir : string
        x : dict or list of dicts
        y : dict or list of dicts
        transforms : list of dicts

        Example
        -------
        >>> from nitrain.datasets import FolderDataset, readers
        >>> from nitrain import transforms as tx
        >>> dataset = FolderDataset('~/Desktop/openneuro/ds004711', 
                                    inputs={'pattern': '{id}/anat/*T1w.nii.gz', 
                                        'exclude': '**run-02*'},
                                    outputs={'file':'participants.tsv', 'column':'age', 
                                       'id': 'participant_id'})
        >>> dataset = FolderDataset('~/Desktop/openneuro/ds004711', 
                        inputs={
                            'input1': readers.PatternReader(
                                'pattern': '{id}/anat/*T1w.nii.gz', 
                                'exclude': '**run-02*')
                            },
                            'input2': readers.PatternReader(
                                'pattern': '{id}/anat/*T1w.nii.gz', 
                                'exclude': '**run-02*')
                            },
                        },
                        outputs = readers.ColumnReader('file':'participants.tsv', 
                                'column':'age', 
                                'id': 'participant_id'),
                        transforms=[
                            {'input2': tx.Resample((128,128,128))}
                            {['input1','input2']: tx.RangeNormalize(0,1)}
                        ]
            )
        """
        if base_dir.startswith('~'):
            base_dir = os.path.expanduser(base_dir)
        
        x_reader = infer_reader(x, base_dir)
        y_reader = infer_reader(y, base_dir)
        x_reader, y_reader = align_readers(x_reader, y_reader)

        self.x_reader = x_reader
        self.y_reader = y_reader
        self.x = x_reader.values
        self.y = y_reader.values
   
        self.base_dir = base_dir
        self.transforms = transforms
    
    def to_platform(self, name, token=None):
        """
        Upload a FolderDataset to the nitrain.dev platform.
        
        This will upload all data from the folder dataset configuration onto
        the nitrain.dev platform. It will be visible at nitrain.dev/datasets. An
        uploaded dataset can then be accessed in the future using the PlatformDataset
        class.
        
        This function is called automatically when you fit a PlatformTrainer on a 
        FolderDataset. In that case, however, the dataset will be deleted after
        training is done if you do not specify it to be cached.
        
        Note that you must have an account at nitrain.dev and also set the
        `NITRAIN_API_KEY` environemnt variable here to use this function. 
        
        Arguments
        ---------
        name : string
            The name of the dataset on the platform.
        
        Examples
        --------
        >>> from nitrain.datasets import FolderDataset
        >>> dataset = FolderDataset('~/Desktop/ds-mini/',
        ...                        x={'pattern':'{id}/anat/*.nii.gz'},
        ...                        y={'file':'participants.tsv', 'id': 'participant_id', 'column': 'age'})
        >>> dataset.to_platform('ds-mini')
        """
        if token is None:
            token = os.environ.get('NITRAIN_API_TOKEN')
            if token is None:
                raise Exception('No api token given or found. Set `NITRAIN_API_TOKEN` or create an account to get your token.')

        # this will raise exception if token is not valid
        user = platform._get_user_from_token(token)
        path = f'{user}/{name}'
        print(f'Uploading dataset to {path}')
        #platform_dataset = platform._convert_to_platform_dataset(self, path)

    def __str__(self):
        return f'FolderDataset with {len(self.x)} records'

    def __repr__(self):
        # TODO: implement transforms __repr__
        if self.transforms:
            tx_str = 'transforms=transforms'
            
        if self.transforms is not None:
            text = f"""FolderDataset(base_dir = '{self.base_dir}',
                    x = {self._x_arg},
                    y = {self._y_arg},
                    {tx_str})"""
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
            transforms=self.transforms
        )
    
