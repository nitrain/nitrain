import os
import textwrap
import tempfile
import logging

from ..platform import (_upload_dataset_to_platform, 
                        _upload_file_to_platform,
                        _upload_job_script_to_platform,
                        _launch_job_on_platform,
                        _check_job_status_on_platform,
                        _convert_to_platform_dataset, 
                        _get_user_from_token)

class CloudTrainer:
    """
    The CloudTrainer class lets you train deep learning models
    on GPU resources in the cloud.
    """
    
    def __init__(self, model, task, name, resource='gpu-small', token=None):
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
        # must have token to use cloud trainer
        if token is None:
            token = os.environ.get('NITRAIN_API_TOKEN')
            if token is None:
                raise Exception('No api token given or found. Set `NITRAIN_API_TOKEN` or create an account to get your token.')

        # this will raise exception if token is not valid
        user = _get_user_from_token(token)
        
        self.user = user
        self.model = model
        self.framework = 'keras' # todo: infer from model
        self.task = task
        self.name = name
        self.resource = resource
        self.token = token
        
        
    def fit(self, loader, epochs):
        """
        Launch a training job on the platform.
        
        This function is used in the same was as for `ModelTrainer`, except that
        calling `fit()` with a `CloudTrainer` will launch a training job on the platform.
        
        If the dataset for the loader passed into this function is not a `PlatformDataset` then the
        dataset will be temporarily uploaded to the cloud for training and then deleted after. To save
        time on repeated training jobs, the loader can be cached by setting `cache=True` when 
        initializing the trainer.
        
        Arguments
        ---------
        loader : an instance of DatasetLoader or similar class
            The batch generator used to train the mode
            
        epochs : integer
            The number of epochs to train the model for.
            
        Returns
        -------
        None. The status of the job can be checked by calling `trainer.status` and the
        fitted model can be eventually retrieved by calling `trainer.model`.
        """
        # TODO: add timestamp to all files, dirs, names, etc
        
        job_name = f'{self.user}__{self.name}'
        job_dir = f'{self.user}/{self.name}'
        
        # Generate training script
        
        # imports
        repr_imports = '''
        from nitrain import datasets, loaders, models, samplers, trainers, transforms as tx
        '''
        
        # dataset
        platform_dataset = _convert_to_platform_dataset(loader.dataset, job_dir)
        repr_dataset = f'''
        dataset = {repr(platform_dataset)}
        '''
        
        # loader
        repr_loader = f'''
        loader = {repr(loader)}
        '''
        
        # model
        repr_model = f'''
        model = models.load("/gcs/ants-dev/models/{self.user}/untrained__{self.name}")
        '''
        
        # trainer
        repr_trainer = f'''
        trainer = trainers.ModelTrainer(model=model, task="{self.task}")
        trainer.fit(loader, epochs={epochs})
        '''
        
        # save model
        repr_save = f'''
        trainer.save("/gcs/ants-dev/models/{job_dir}")
        '''
        
        # write training script to file
        script_file = tempfile.NamedTemporaryFile(suffix=f'{job_name}.py')
        with open(script_file.name, 'w') as f:
            f.write(textwrap.dedent(repr_imports))
            f.write(textwrap.dedent(repr_dataset))
            f.write(textwrap.dedent(repr_loader))
            f.write(textwrap.dedent(repr_model))
            f.write(textwrap.dedent(repr_trainer))
            f.write(textwrap.dedent(repr_save))
        
        # upload training script to platform: /ants-dev/code/{user}/{name}.py
        print('Uploading training script...')
        _upload_job_script_to_platform(script_file, self.name)
        
        ## upload original dataset to platform: /ants-dev/datasets/{user}/{name}/
        #print('Uploading dataset...')
        #_upload_dataset_to_platform(loader.dataset, self.name)
        #
        ## upload untrained model to platform: /ants-dev/models/{user}/{name}.keras
        #print('Uploading model...')
        #model_file = tempfile.NamedTemporaryFile(suffix='.keras')
        #self.save(model_file.name)
        #_upload_file_to_platform(model_file, 'models', f'untrained__{self.name}.keras')
        
        # launch job
        #_launch_job_on_platform(self.name)
        
    def save(self, filename):
        if self.framework == 'keras':
            if not filename.endswith('.keras'):
                filename = filename + '.keras'
            self.model.save(filename)
            
    @property
    def status(self):
        """
        Check status of launched training job
        """
        return _check_job_status_on_platform(self.name)

