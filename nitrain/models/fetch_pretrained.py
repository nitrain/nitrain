

def fetch_pretrained(name, cache_dir=None):
    """
    Fetch a pretrained model from ANTsPyNet. Pretrained
    models can be used to make predictions (inference) on
    your data or as a starting point for fine-tuning to help
    improve model fitting on your data.
    
    Returns
    -------
    An instance of the PretrainedModel class that includes many
    convenience functions. The actual model can be accessed via
    the .model property.
    """
    import antspynet
    weights = antspynet.get_pretrained_network(name, 
                                               antsxnet_cache_directory=cache_dir)
