# loaders help you feed your data into deep learning models with transforms
import math
import numpy as np
import torch
import ants
    
from ..datasets.transform_dataset import TransformDataset

class TorchLoader(torch.utils.data.DataLoader):
    
    def __init__(self, dataset, batch_size, shuffle=False, x_transforms=None, y_transforms=None, co_transforms=None, ):
        transform_dataset = TransformDataset(dataset, x_transforms=x_transforms)
        super(TorchLoader, self).__init__(transform_dataset, batch_size=batch_size, shuffle=shuffle)