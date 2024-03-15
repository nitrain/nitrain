


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
    >>> trainer = ModelTrainer(model)
    >>> trainer.fit(loader, epochs=10)
    """
    
    def __init__(self, model):
        self.model = model
        
    def fit(self, loader, epochs):
        pass