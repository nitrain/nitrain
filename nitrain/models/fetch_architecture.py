
from inspect import getmembers, isfunction

def fetch_architecture(name, dim=None):
    """
    Fetch an architecture function based on its name and input image 
    dimensions (2, 3, or None).
    
    Arguments
    ---------
    name string
        One of the available architectures to be fetched. Available
        architectures are those with functions called 'create_{name}_model()'.
    
    dim integer (2 or 3)
        Whether to pull the 2-dimensional or 3-dimensional version of the architecture
        function. Can be left as None if there is no 2d vs 3d model (e.g., for autoencoders)
    Returns
    -------
    A filename string
    Example
    -------
    >>> from nitrain import models
    >>> vgg_fn = models.fetch_architecture('vgg', dim=3)
    >>> vgg_model = vgg_fn((128, 128, 128, 1))
    >>> autoencoder_fn = models.fetch_architecture('autoencoder')
    >>> autoencoder_model = autoencoder_fn((784, 500, 500, 2000, 10))
    """
    import antspynet
    
    try:
        if dim is not None:
            fstr = f'create_{name}_model_{dim}d'
            fn = getattr(antspynet.architectures, fstr)
        else:
            fstr = f'create_{name}_model'
            fn = getattr(antspynet.architectures, fstr)
    except AttributeError:
        raise ValueError(f'Architecture function {fstr} does not exist.')
    return fn


def list_architectures():
    """
    List all available architectures and their context-of-use.
    
    Arguments
    ---------
    N/A
    
    Returns
    -------
    A list of 2-item string list where each item is the name of an 
    architecture and an associated image dimension ('' for None, '2d', or '3d').
    So a result of ['alexnet', '2d'] means that you can create an architecture
    function via 'create_{alexnet}_model_{2d}'
    
    Example
    -------
    >>> from nitrain import models
    >>> models.list_architectures()    
    """
    import antspynet
    archs = [f[0].split('create_')[1].split('_model_') for f in getmembers(antspynet.architectures, isfunction) if f[0].startswith('create_')]
    # add empty string for non-dimensioned models just to be consistent
    def add_empty(x):
        if len(x) == 1:
            x.append('')
        return x
    archs = [add_empty(arch) for arch in archs]
    return archs