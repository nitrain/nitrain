
class Trainer:
    """
    The Trainer class provides a general high-level inference to train
    models from multiple frameworks on nitrain data loaders.
    """
    
    def __init__(self, 
                 model,
                 task=None,
                 optimizer=None,
                 loss=None,
                 metrics=None,
                 **kwargs):
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
        elif task in ('classification', 'segmentation'):
            optimizer = 'adam' if optimizer is None else optimizer
            if model.output_shape[-1] > 1:
                loss = 'categorical_crossentropy' if loss is None else loss
            else:
                loss = 'binary_crossentropy' if loss is None else loss
            metrics = ['accuracy'] if metrics is None else metrics
        elif task is None:
            # TODO: ensure optimizer and loss is supplied
            if optimizer is None and loss is None:
                raise Exception('If task is None then optimizer and loss must be supplied.')
            pass
        else:
            raise Exception('Valid tasks: `regression`, `segmentation`, `classification`.')
        
        self.task = task
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
                
        framework = infer_framework(model)
        self.framework = framework
        
        if self.framework == 'keras':
            self.model.compile(optimizer=self.optimizer,
                               loss=self.loss,
                               metrics=self.metrics)
            
        self.kwargs = kwargs
        
    def fit(self, loader, epochs, validation=None, **kwargs):
        from .torch_utils import torch_model_fit
        
        if self.framework == 'keras':
            if type(loader).__name__ == 'Loader':
                loader = loader.to_keras()
            return self.model.fit(loader, epochs=epochs, **kwargs)
        elif self.framework == 'torch':
            return torch_model_fit(self, loader, epochs, validation, **kwargs)

    def evaluate(self, loader):
        if self.framework == 'keras':
            if type(loader).__name__ == 'Loader':
                loader = loader.to_keras()
            return self.model.evaluate(loader)
    
    def predict(self, loader):
        if self.framework == 'keras':
            if type(loader).__name__ == 'Loader':
                loader = loader.to_keras()
            return self.model.predict(loader)
    
    def summary(self):
        if self.framework == 'keras':
            return self.model.summary()
    
    def save(self, path):
        if self.framework == 'keras':
            self.model.save(path)
    
    def __repr__(self):
        s = 'Trainer ({})\n'.format(self.task)
        s = s +\
            '     {:<10} : {}\n'.format('Framework', self.framework)+\
            '     {:<10} : {}\n'.format('Loss', self.loss)+\
            '     {:<10} : {}\n'.format('Optimizer', self.optimizer)+\
            '     {:<10} : {}\n'.format('Metrics', self.metrics)
        return s


def infer_framework(model):
    model_type = str(type(model))
    if 'keras' in model_type:
        return 'keras'
    
    if 'torch' in model_type:
        return 'torch'
    
    if 'monai' in model_type:
        return 'torch'
    
    raise ValueError('Could not infer framework from model')
    