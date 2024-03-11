import keras
import numpy as np
import math

from ..datasets.random_transform_dataset import RandomTransformDataset

class KerasLoader(keras.utils.PyDataset):

    def __init__(self, dataset, batch_size, x_transforms=None, y_transforms=None, co_transforms=None, **kwargs):
        super().__init__(**kwargs)
        transform_dataset = RandomTransformDataset(dataset, x_transforms=x_transforms)
        self._dataset = transform_dataset
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