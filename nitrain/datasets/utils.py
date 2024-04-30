import os
import sys

import pandas as pd
from tempfile import mkdtemp
import ntimage as nti

from ..utils import get_nitrain_dir
    
def fetch_data(name, path=None):
    """
    Download example datasets from OpenNeuro and other sources.
    Datasets are pulled using `datalad` so the raw data will 
    not actually be dowloaded until it is needed. This makes it
    really fast.
    
    Arguments
    ---------
    name : string
        the dataset to download
        Options:
            - ds004711 [OpenNeuroDatasets/ds004711]
            - example/t1-age
            - example/t1-t1_mask
            
    Example
    -------
    import nitrain as nt
    ds = nt.fetch_data('openneuro/ds004711')
    """
    
    if path is None:
        path = get_nitrain_dir()
    
    if name.startswith('openneuro'):
        import datalad.api as dl
        
        save_dir = os.path.join(path, name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        # load openneuro dataset using datalad
        res = dl.clone(source=f'///{name}', path=save_dir)
    else:
        raise ValueError('Dataset name not recognized.')

    return save_dir


def reduce_to_list(d, idx=0):
    result = []
    for key, val in d.items():
        if isinstance(val, dict):
            result.extend([reduce_to_list(d[key], idx+1)])
        else:
            result.append(val)
        
    return result if len(result) > 1 else result[0]


def apply_transforms(tx_name, tx_value, inputs, outputs, force=False):
    """
    Apply transforms recursively based on match between transform name
    and input label. If there is a match and the input is nested, then all
    children of that input will receive the transform.
    """
    if not isinstance(tx_name, tuple):
        tx_name = (tx_name,)
    if not isinstance(tx_value, list):
        if isinstance(tx_value, tuple):
            tx_value = list(tx_value)
        else:
            tx_value = [tx_value]

    if inputs:
        for input_name, input_value in inputs.items():
            if isinstance(input_value, dict):
                if input_name in tx_name:
                    apply_transforms(tx_name, tx_value, input_value, None, force=True)
                else:
                    apply_transforms(tx_name, tx_value, input_value, None, force=force)
            else:
                if (input_name in tx_name) | force:
                    for tx_fn in tx_value:
                        inputs[input_name] = tx_fn(inputs[input_name])

    if outputs:
        for output_name, output_value in outputs.items():
            if isinstance(output_value, dict):
                if output_name in tx_name:
                    apply_transforms(tx_name, tx_value, None, output_value, force=True)
                else:
                    apply_transforms(tx_name, tx_value, None, output_value, force=force)
            else:
                if (output_name in tx_name) | force:
                    for tx_fn in tx_value:
                        outputs[output_name] = tx_fn(outputs[output_name])
    
    return inputs, outputs
            