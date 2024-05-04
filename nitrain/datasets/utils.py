import os
import sys
import shutil
import numpy as np
import pandas as pd
from tempfile import mkdtemp
import ntimage as nti

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

            img2d = nti.ones((30,40))
            img3d = nti.ones((30,40,50))
            img3d_seg = nti.zeros_like(img3d).astype('uint8')
            img3d_seg[10:20,10:30,10:40] = 1
            
            img3d_multiseg = nti.zeros_like(img3d).astype('uint8')
            img3d_multiseg[:20,:20,:20] = 1
            img3d_multiseg[20:30,20:30,20:30] = 2
            img3d_multiseg[30:,30:,30:]=0
            
            img3d_large = nti.ones((60,80,100))
            for i in range(10):
                sub_dir = os.path.join(save_dir, f'sub_{i}')
                os.mkdir(sub_dir)
                nti.save(img2d + i, os.path.join(sub_dir, 'img2d.nii.gz'))
                nti.save(img3d + i, os.path.join(sub_dir, 'img3d.nii.gz'))
                nti.save(img3d_large + i, os.path.join(sub_dir, 'img3d_large.nii.gz'))
                nti.save(img3d + i + 100, os.path.join(sub_dir, 'img3d_100.nii.gz'))
                nti.save(img3d_seg, os.path.join(sub_dir, 'img3d_seg.nii.gz'))
                nti.save(img3d_multiseg, os.path.join(sub_dir, 'img3d_multiseg.nii.gz'))
                nti.save(img3d + i + 1000, os.path.join(sub_dir, 'img3d_1000.nii.gz'))
            
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
            