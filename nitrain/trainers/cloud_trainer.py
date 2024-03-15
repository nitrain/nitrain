import os

class CloudTrainer:
    """
    The CloudTrainer class lets you train deep learning models
    on GPU resources in the cloud via the ants.dev platform. All
    training jobs can be managed at ants.dev.
    
    
    Examples
    --------
    >>> download = fetch_data('openneuro/ds004711')
    >>> data = FolderDataset(download.path)
    >>> loader = DatasetLoader(data, batch_size=32)
    >>> model_fn = fetch_architecture('autoencoder')
    >>> model = model_fn((120, 60, 30))
    >>> trainer = CloudTrainer(model, name='t1-brain-age', resource='gpu-small')
    >>> job = trainer.fit(loader, epochs=10)
    >>> print(job.status)
    >>> print(job.fitted_model)
    """
    
    def __init__(self, model, name, resource='gpu-small', api_token=None):
        
        # check for platform credentials
        if api_token is None:
            api_token = os.environ.get('ANTS_DEV_TOKEN')
            if api_token is None:
                raise Exception('No api token given or found. Set `ANTS_DEV_TOKEN` or create an account at https://www.ants.dev/sign-up to get your token.')
        
        self.model = model
        self.name = name
        self.resource = resource
    
    def fit(self, loader, epochs):
        
        # generate training script
        
        # upload data 
        pass