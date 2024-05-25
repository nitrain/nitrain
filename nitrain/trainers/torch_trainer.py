import torch
from .torch_utils import torch_trainer_fit, torch_trainer_evaluate


class TorchTrainer:
    """
    The TorchTrainer class provides high-level functionality to train
    models from pytorch on nitrain data loaders.
    """

    def __init__(self,
                 model,
                 optimizer,
                 loss,
                 metrics,
                 device='cpu',
                 **kwargs):
        """
        Initialize a trainer from a pytorch model
        """
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.device = device
        self.kwargs = kwargs
        
    def fit(self, loader, epochs, validation=None, **kwargs):
        return torch_trainer_fit(self.model, self.loss, self.optimizer, 
                                 self.metrics, self.device, loader, 
                                 epochs, validation, **kwargs)

    def evaluate(self, loader):
        return torch_trainer_evaluate(self.model, self.metrics, self.device, loader)
    
    def predict(self, loader):
        pass
    
    def summary(self):
        pass
    
    def save(self, path):
        pass
    
    def __repr__(self):
        s = 'TorchTrainer ({})\n'.format('Custom')
        s = s +\
            '     {:<10} : {}\n'.format('Loss', self.loss)+\
            '     {:<10} : {}\n'.format('Optimizer', self.optimizer)+\
            '     {:<10} : {}\n'.format('Metrics', self.metrics)+\
            '     {:<10} : {}\n'.format('Device', self.device)
        return s

    