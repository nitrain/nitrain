


class ModelTrainer:
    """
    The ModelTrainer class is a high-level tool for training
    a deep learning model on your data.
    
    Examples
    --------
    >>> download = fetch_data('openneuro/ds004711')
    >>> data = FolderDataset(download.path)
    >>> loader = DatasetLoader(data, batch_size=32)
    >>> model_fn = fetch_architecture('autoencoder')
    >>> model = model_fn((120, 60, 30))
    >>> trainer = ModelTrainer(model)
    >>> trainer.compile(optimizer='Adam')
    >>> trainer.fit_loader(loader)
    """
    
    def __init__(self, model):
        self.model = model
        
    def fit_loader(self, data):
        pass