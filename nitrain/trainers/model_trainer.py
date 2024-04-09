


class ModelTrainer:
    """
    The ModelTrainer class provides high-level functionality to train
    deep learning models on dataset loaders. It wraps the most popular
    frameworks under a common interface.
    
    Examples
    --------
    >>> download = fetch_data('openneuro/ds004711')
    >>> data = FolderDataset(download.path)
    >>> loader = DatasetLoader(data, batch_size=32)
    >>> model_fn = fetch_architecture('autoencoder')
    >>> model = model_fn((120, 60, 30))
    >>> trainer = ModelTrainer(model, task='regression')
    >>> trainer.fit(loader, epochs=10)
    """
    
    def __init__(self, 
                 model,
                 task,
                 optimizer=None,
                 loss=None,
                 metrics=None):
        """
        Create a model trainer with sensible defaults or user-defined
        settings from a pytorch, keras, or tensorflow model.
        
        Arguments
        ---------
        task: string
            options: regression, classification
            
        ## callbacks
        ## regularizers
        ## initializers
        ## constraints
        ## metrics
        ## losses
        """
        self.model = model
        if task == 'regression':
            optimizer = 'adam' if optimizer is None else optimizer
            loss = 'mse' if loss is None else loss
            metrics = ['mse'] if metrics is None else metrics
        elif task == 'classification':
            optimizer = 'adam' if optimizer is None else optimizer
            if model.output_shape[-1] == 1:
                loss = 'categorical_crossentropy' if loss is None else loss
            else:
                loss = 'binary_crossentropy' if loss is None else loss
            metrics = ['accuracy'] if metrics is None else metrics
        else:
            raise ValueError('The only valid tasks are `regression` and `classification`.')
        
        self.task = task
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
                
        framework = infer_framework(model)
        self.framework = framework
        
        if framework == 'keras':
            self.model.compile(optimizer='adam',
                               loss='mse',
                               metrics=['mse'])
        
    def fit(self, loader, epochs, **kwargs):
        if self.framework == 'keras':
            if type(loader).__name__ == 'DatasetLoader':
                loader = loader.to_keras()
            return self.model.fit(loader, epochs=epochs, **kwargs)

    def evaluate(self, loader):
        if self.framework == 'keras':
            if type(loader).__name__ == 'DatasetLoader':
                loader = loader.to_keras()
            return self.model.evaluate(loader)
    
    def predict(self, loader):
        if self.framework == 'keras':
            if type(loader).__name__ == 'DatasetLoader':
                loader = loader.to_keras()
            return self.model.predict(loader)
    
    def summary(self):
        if self.framework == 'keras':
            return self.model.summary()
    
    def save(self, path):
        if self.framework == 'keras':
            self.model.save(path)


def infer_framework(model):
    model_type = str(type(model))
    if 'keras' in model_type:
        return 'keras'
    if 'torch' in model_type:
        return 'torch'
    