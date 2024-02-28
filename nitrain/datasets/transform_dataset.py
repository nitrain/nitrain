import torch
import numpy as np

class TransformDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset, x_transforms=None):
        self.dataset = dataset
        self.x_transforms = x_transforms
        
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        
        # apply random transforms here
        if self.x_transforms is not None:
            if isinstance(x, list):
                x = [self.x_transforms(xx) for xx in x]
            else:
                x = self.x_transforms(x)
        
        if isinstance(x, list):
            return np.array([np.expand_dims(xx.numpy(), -1) for xx in x]), y
        else:
            return np.expand_dims(x.numpy(), -1), y

    def __len__(self):
        return len(self.dataset)