import os
import textwrap

from ..datasets.utils_platform import _convert_to_platform_dataset

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
        
        self.user = 'nick-2'
        self.model = model
        self.task = task
        self.name = name
        self.resource = resource
        self.api_token = api_token
    
    def fit(self, loader, epochs):
        """
        Launch a training job in the cloud
        """
        # Generate training script
        
        # imports
        repr_imports = '''
        from nitrain import datasets, loaders, trainers, transforms as tx
        '''
        
        # dataset
        platform_dataset = _convert_to_platform_dataset(loader.dataset, f'{self.user}/{self.name}')
        repr_dataset = f'''
        dataset = {repr(platform_dataset)}
        '''
        
        # loader
        repr_loader = f'''
        loader = {repr(loader)}
        '''
        
        # model
        repr_model = f'''
        model = utils.fetch_model_on_platform("{self.user}/{self.name}__model")
        '''
        
        # - trainer
        repr_trainer = f'''
        trainer = ModelTrainer(model=model, task={self.task})
        trainer.fit(epochs={epochs})
        '''
        
        # write training script to file
        with open(f'/Users/ni5875cu/Desktop/{self.name}.py', 'w') as f:
            f.write(textwrap.dedent(repr_imports))
            f.write(textwrap.dedent(repr_dataset))
            f.write(textwrap.dedent(repr_loader))
            f.write(textwrap.dedent(repr_model))
            f.write(textwrap.dedent(repr_trainer))
        
        # upload training script to platform
        
        # upload original dataset to platform
        
        # upload untrained model to platform
        
        # launch job
        
    
    @property
    def status(self):
        """
        Check status of launched training job
        """
        pass