import math
import numpy as np

from ..datasets.random_transform_dataset import RandomTransformDataset

class DatasetLoader:
    
    def __init__(self, 
                 dataset, 
                 batch_size, 
                 x_transforms=None, 
                 y_transforms=None, 
                 co_transforms=None,
                 expand_dim=-1,
                 shuffle=False):
        """
        Arguments
        ---------
        
        Examples
        --------
        ds = Dataset()
        ld = DatasetLoader(ds)
        xb, yb = next(iter(ld))

        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.expand_dim = expand_dim
        self.x_transforms = x_transforms

    def __iter__(self):
        batch_size = self.batch_size
        dataset = self.dataset
        n_batches = math.ceil(len(dataset) / batch_size)
              
        batch_idx = 0
        while batch_idx < n_batches:
            data_indices = slice(batch_idx*batch_size, min((batch_idx+1)*batch_size, len(dataset)))
            x, y = dataset[data_indices]
           
            # perform random transforms
            if self.x_transforms:
                for tx_fn in self.x_transforms:
                    x = [tx_fn(img) for img in x]
          
            # gather slices
       
            # shuffle images / slices
           
            if self.expand_dim is not None:
                x_arr = np.array([np.expand_dims(xx.numpy(), self.expand_dim) for xx in x])
            else:
                x_arr = np.array([xx.numpy() for xx in x])

            yield x_arr, y
            batch_idx += 1

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)
    

