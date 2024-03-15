import math
import numpy as np

import keras

from .. import samplers

# Keras v3 only - for Keras v2 use DatasetLoader().to_keras()
if int(keras.__version__.split('.')[0]) == 3:
    
    class KerasLoader(keras.utils.PyDataset):

        def __init__(self, dataset, batch_size, x_transforms=None, y_transforms=None, co_transforms=None, **kwargs):
            super().__init__(**kwargs)
            self._dataset = dataset
            self.dataset = dataset
            self.batch_size = batch_size

        def __getitem__(self, idx):
            batch_size = self.batch_size
            dataset = self._dataset

            data_indices = slice(idx*batch_size, min((idx+1)*batch_size, len(dataset)))
            x, y = dataset[data_indices]
            return x, y
        
        def __len__(self):
            return math.ceil(len(self.dataset) / self.batch_size)