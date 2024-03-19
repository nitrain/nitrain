import os

class CloudTrainer:
    """
    The CloudTrainer class lets you train deep learning models
    on GPU resources in the cloud.
    """
    
    def __init__(self, model, task, name, resource='gpu-small', api_token=None):
        """
        Initialize a cloud trainer
        
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
        >>> print(job.model)
        """
        
        # check for platform credentials
        if api_token is None:
            api_token = os.environ.get('NITRAIN_API_TOKEN')
            if api_token is None:
                raise Exception('No api token given or found. Set `NITRAIN_API_TOKEN` or create an account to get your token.')
        
        self.model = model
        self.task = task
        self.name = name
        self.resource = resource
        self.api_token = api_token
    
    def fit(self, loader, epochs):
        """
        Launch a training job in the cloud
        """
        # generate training script
        # - data
        #   - 
        # - loader
        # - model
        # - trainer
        
        
        # upload data 
        pass
    
    @property
    def status(self):
        """
        Check status of launched training job
        """
        pass