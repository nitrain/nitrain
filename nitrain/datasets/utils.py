import os
import sys
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


class SilentFunction(object):
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()