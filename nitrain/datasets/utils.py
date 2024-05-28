import os
import sys
import shutil
import numpy as np
import pandas as pd
from tempfile import mkdtemp
import ants

from ..utils import get_nitrain_dir
    
def fetch_data(name, path=None, overwrite=False):
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
    else:
        path = os.path.expanduser(path)
    
    if name.startswith('openneuro'):
        import datalad.api as dl
        
        save_dir = os.path.join(path, name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        # load openneuro dataset using datalad
        res = dl.clone(source=f'///{name}', path=save_dir)
    elif name.startswith('example'):
        if name == 'example-01':
            # folder with nifti images and csv file
            # this example is good for testing and sanity checks
            # set up directory
            save_dir = os.path.join(path, name)
            
            if overwrite:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            else:
                if os.path.exists(save_dir):
                    return save_dir
            
            os.makedirs(save_dir, exist_ok=True)

            img2d = ants.from_numpy(np.ones((30,40)))
            img3d = ants.from_numpy(np.ones((30,40,50)))
            img3d_seg = ants.from_numpy(np.zeros(img3d.shape).astype('uint8'))
            img3d_seg[10:20,10:30,10:40] = 1
            
            img3d_multiseg = ants.from_numpy(np.zeros(img3d.shape).astype('uint8'))
            img3d_multiseg[:20,:20,:20] = 1
            img3d_multiseg[20:30,20:30,20:30] = 2
            img3d_multiseg[30:,30:,30:]=0
            
            img3d_large = ants.from_numpy(np.ones((60,80,100)))
            for i in range(10):
                sub_dir = os.path.join(save_dir, f'sub_{i}')
                os.mkdir(sub_dir)
                ants.image_write(img2d + i, os.path.join(sub_dir, 'img2d.nii.gz'))
                ants.image_write(img3d + i, os.path.join(sub_dir, 'img3d.nii.gz'))
                ants.image_write(img3d_large + i, os.path.join(sub_dir, 'img3d_large.nii.gz'))
                ants.image_write(img3d + i + 100, os.path.join(sub_dir, 'img3d_100.nii.gz'))
                ants.image_write(img3d_seg, os.path.join(sub_dir, 'img3d_seg.nii.gz'))
                ants.image_write(img3d_multiseg, os.path.join(sub_dir, 'img3d_multiseg.nii.gz'))
                ants.image_write(img3d + i + 1000, os.path.join(sub_dir, 'img3d_1000.nii.gz'))
            
            # write csv file
            ids = [f'sub_{i}' for i in range(10)]
            age = [i + 50 for i in range(10)]
            weight = [i + 200 for i in range(10)]
            img2d = [f'sub_{i}/img2d.nii.gz' for i in range(10)]
            img3d = [f'sub_{i}/img3d.nii.gz' for i in range(10)]
            img3d_large = [f'sub_{i}/img3d_large.nii.gz' for i in range(10)]
            img3d_100 = [f'sub_{i}/img3d_100.nii.gz' for i in range(10)]
            img3d_1000 = [f'sub_{i}/img3d_1000.nii.gz' for i in range(10)]
            img3d_seg = [f'sub_{i}/img3d_seg.nii.gz' for i in range(10)]
            df = pd.DataFrame({'sub_id': ids, 'age': age, 'weight': weight,
                               'img2d': img2d, 'img3d': img3d, 'img3d_large': img3d_large,
                               'img3d_100':img3d_100, 'img3d_1000': img3d_1000,
                               'img3d_seg': img3d_seg})
            df.to_csv(os.path.join(save_dir, 'participants.csv'), index=False)
            
    else:
        raise ValueError('Dataset name not recognized.')

    return save_dir


def reduce_to_list(d, idx=0):
    result = []
    for k, v in d.items():
        if isinstance(v, dict):
            result.append(reduce_to_list(v))
        else:
            result.append(v)
    return result if len(result) > 1 else result[0]

def retrieve_values_from_dict(d, names):
    values = []
    for k, v in d.items():
        if isinstance(v, dict):
            values.extend(retrieve_values_from_dict(v, names))
        if k in names:
            if isinstance(v, dict):
                values.extend(reduce_to_list(v))
            else:
                values.append(v)
    return list(values)

def overwrite_values_in_dict(d, names, new_values):
    for k, v in d.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if k2 in names:
                    d[k][k2] = new_values[k2]
        else:
            if k in names:
                d[k] = new_values[k]
    return d
    
def apply_transforms(tx_name, tx_value, inputs, outputs):
    if not isinstance(tx_name, tuple):
        tx_name = (tx_name,)
        
    if not isinstance(tx_value, list):
        if isinstance(tx_value, tuple):
            tx_value = list(tx_value)
        else:
            tx_value = [tx_value]
            
    # first, get all inputs and outputs that match tx_name
    needed_inputs = retrieve_values_from_dict(inputs, tx_name)
    needed_outputs = retrieve_values_from_dict(outputs, tx_name)
    needed_values = list(needed_inputs) + list(needed_outputs)

    if len(needed_values) < len(tx_name):
        raise Exception('Some names in your transform were not found. Check for typos in the key labels.')
    
    # next, apply transforms to all matched inputs / outputs together
    for tx_fn in tx_value:
        needed_values = tx_fn(*needed_values)
        if not isinstance(needed_values, (tuple,list)):
            needed_values = [needed_values]
    
    # finally, overwrite the original inputs / outputs with transformed versions
    new_inputs = overwrite_values_in_dict(inputs, tx_name, {n:nv for n,nv in zip(tx_name, needed_values)})
    new_outputs = overwrite_values_in_dict(outputs, tx_name, {n:nv for n,nv in zip(tx_name, needed_values)})
    return new_inputs, new_outputs
