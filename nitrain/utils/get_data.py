import os
from pathlib import Path
import datalad.api as dl


def get_nitrain_dir():
    downloads_path = str(Path.home() / "Desktop")
    return downloads_path    


def download_data(name, path=None):
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
            - ds003826 [OpenNeuroDatasets/ds003826]
            
    Example
    -------
    >>> from nitrain.utils import download_data
    >>> download_data('ds003826')
    """
    
    if path is None:
        path = get_nitrain_dir()
        print(path)
        
    if name == "ds003826":
        dl.clone(source='///openneuro/ds003826',
                 path=os.path.join(path, 'ds003826'))
        pass    
    