# loaders help you feed your data into deep learning models with transforms
import math
import numpy as np
import torch
import ants

class TransformDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset, x_transforms=None):
        self.dataset = dataset
        self.x_transforms = x_transforms
        
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        
        # apply random transforms here
        if self.x_transforms is not None:
            x = self.x_transforms(x)
            
        return np.expand_dims(x.numpy(), -1), y

    def __len__(self):
        return len(self.dataset)
    

class TorchLoader(torch.utils.data.DataLoader):
    
    def __init__(self, dataset, batch_size, shuffle=False, x_transforms=None, y_transforms=None, co_transforms=None, ):
        transform_dataset = TransformDataset(dataset, x_transforms=x_transforms)
        super(TorchLoader, self).__init__(transform_dataset, batch_size=batch_size, shuffle=shuffle)