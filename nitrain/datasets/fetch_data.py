import os
import datalad.api as dl

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
            
    Example
    -------
    >>> from nitrain.utils import download_data
    >>> ds = fetch_data('openneuro/ds004711')
    """
    
    if path is None:
        path = get_nitrain_dir()
    
    save_dir = os.path.join(path, name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    ref = dl.clone(source=f'///{name}', path=save_dir)
    return ref


