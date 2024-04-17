import glob
import os
from parse import parse
from fnmatch import fnmatch

import datalad.api as dl
import pandas as pd
import numpy as np
import ntimage as nt

def infer_reader(x, base_dir=None):
    """
    Infer reader from user-supplied values.
    
    A reader is a possible value to the `x` and `y` arguments of any nitrain dataset class. The
    possible values include the following:
    
    - numpy array (ArrayReader)
    - list of ntimages (ImageReader)
    - dict containing glob patterns to read images from file (PatternReader)
    - dict containing BIDS entities to read images from file in BIDS folder (BIDSReader)
    - dict containing info to read columns from csv files (ColumnReader)
    - dict containing info to read images from file in google cloud storage (GoogleCloudReader)
    - dict containing info to read images from file hosted on nitrain.dev (PlatformReader)
    - list with a combination of any of the above readers (ComposeReader)
    
    Note that when a reader is created, it is actually checked for validity -- i.e., that data
    can be served from it. For example, creating a reader to read images from files that do not
    exist will raise an exception.
    
    Examples
    --------
    >>> base_dir = os.path.expanduser('~/Desktop/openneuro/ds004711')
    >>> array = np.random.normal(40,10,(10,50,50,50))
    >>> x = infer_config(array)
    >>> x = infer_config([nt.load(nt.example_data('r16')) for _ in range(10)])
    >>> x = infer_config([{'pattern': '{id}/anat/*.nii.gz'}, {'pattern': '{id}/anat/*.nii.gz'}], base_dir) 
    >>> x = infer_config({'pattern': '{id}/anat/*.nii.gz'}, base_dir) 
    >>> x = infer_config({'pattern': '*/anat/*.nii.gz'}, base_dir)
    >>> x = infer_config({'pattern': '**/*T1w*'}, base_dir) 
    >>> x = infer_config({'file': 'participants.tsv', 'column': 'age', 'id': 'participant_id'}, base_dir) 
    >>> x = infer_config({'file': 'participants.tsv', 'column': 't1', 'image': True}, base_dir) 
    """
    reader = None
    if isinstance(x, list):
        # list of ntimages
        if isinstance(x[0], nt.NTImage):
            return ImageReader(x)
        # list of multiple ()potentially mixed) readers
        elif isinstance(x[0], dict):
            readers = [infer_reader(reader, base_dir=base_dir) for reader in x]
            return ComposeReader(readers)
        # list that is meant to be an array or multiple-images
        elif isinstance(x[0], list):
            if isinstance(x[0][0], nt.NTImage):
                readers = []
                for i in range(len(x[0])):
                    readers.append(ImageReader([xx[i] for xx in x]))
                return ComposeReader(readers)
            else:
                return ArrayReader(np.array(x))    
        else:
            return ArrayReader(np.array(x))
        
    elif isinstance(x, dict):
        # options:
        # - PatternReader
        # - ColumnReader
        # - GoogleCloudReader
        # - PlatformReader
        if 'pattern' in x.keys():
            return PatternReader(base_dir=base_dir, **x)
        if 'file' in x.keys():
            return ColumnReader(base_dir=base_dir, **x)

        # TODO: how to check for BIDSReader
        if 'dataytype' in x.keys():
            return BIDSReader(base_dir=base_dir, **x)
        
    elif isinstance(x, np.ndarray):
        return ArrayReader(x)
    
    if reader is None:
        raise Exception(f'Could not infer a configuration from given value: {x}')
    
    return reader


def align_readers(x, y):
    """
    Align readers based on ID or something other pattern.
    """
    if x.ids is None:
        raise Exception('`x` is missing ids. Specify `{id}` somewhere in the pattern.')
    if y.ids is None:
        if isinstance(y, PatternReader):
            raise Exception('`y` is missing ids. Specify `{id}` somewhere in the pattern.')
        elif isinstance(y, ColumnReader):
            raise Exception('`y` is missing ids. Specify "id": "COL_NAME" in the file dict.')
    
    x_ids = x.ids
    y_ids = y.ids
    
    # match ids
    matched_ids = sorted(list(set(x_ids) & set(y_ids)))
    if len(matched_ids) == 0:
        raise Exception('No matches found between `x` ids and `y` ids. Double check your reader.')
    
    # take only matched ids in x
    keep_idx = [i for i in range(len(x.ids)) if x.ids[i] in matched_ids]
    x.ids = matched_ids
    x.values = [x.values[i] for i in keep_idx]

    # take only matched ids in y
    keep_idx = [i for i in range(len(y.ids)) if y.ids[i] in matched_ids]
    y.ids = matched_ids
    if isinstance(y.values, np.ndarray):
        y.values = np.array([y.values[i] for i in keep_idx])
    else:
        y.values = [y.values[i] for i in keep_idx]

    return x, y

