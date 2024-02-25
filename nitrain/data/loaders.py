# loaders help you feed your data into deep learning models with transforms

class DatasetLoader:
    
    def __init__(self, dataset, x_transforms=None, y_transforms=None, co_transforms=None):
        self.dataset = dataset
    
    @property
    def input_shape(self):
        # load the first input, perform any transforms, and get shape
        return self.dataset[0][0].shape