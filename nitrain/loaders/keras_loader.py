import math
import numpy as np

import tensorflow as tf

from .. import samplers

class KerasLoader(tf.keras.utils.Sequence):
    
    def __init__(self, 
                 dataset, 
                 batch_size, 
                 x_transforms=None, 
                 y_transforms=None, 
                 co_transforms=None,
                 expand_dims=-1,
                 sampler=None):
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
        self.expand_dims = expand_dims
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms
        self.co_transforms = co_transforms
        
        if sampler is None:
            sampler = samplers.BaseSampler()
        self.sampler = sampler
        
    
    def __getitem__(self, index):
        
        batch_size = self.batch_size
        dataset = self.dataset
        data_indices = slice(index*batch_size, min((index+1)*batch_size, len(dataset)))
        x, y = dataset[data_indices]
        
        # sample the batch
        sampled_batch = self.sampler(x, y)
        x_batch, y_batch = next(iter(sampled_batch))
        # a normal sampler will just return the entire (shuffled, if specified) batch once
        # a slice sampler will return shuffled slices with batch size = sampler.batch_size
        #for x_batch, y_batch in sampled_batch:
            
        if self.expand_dims is not None:
            x_batch = np.array([np.expand_dims(xx.numpy(), self.expand_dims) for xx in x_batch])
        else:
            x_batch = np.array([xx.numpy() for xx in x_batch])
        
        return x_batch, y_batch

    def __len__(self):
        return math.ceil((len(self.dataset) / self.batch_size) * len(self.sampler))
    


# keras 3.0
#class Keras3Loader(keras.utils.PyDataset):
#
#    def __init__(self, dataset, batch_size, x_transforms=None, y_transforms=None, co_transforms=None, **kwargs):
#        super().__init__(**kwargs)
#        transform_dataset = RandomTransformDataset(dataset, x_transforms=x_transforms)
#        self._dataset = transform_dataset
#        self.dataset = dataset
#        self.batch_size = batch_size
#
#    def __getitem__(self, idx):
#        batch_size = self.batch_size
#        dataset = self._dataset
#
#        data_indices = slice(idx*batch_size, min((idx+1)*batch_size, len(dataset)))
#        x, y = dataset[data_indices]
#        return x, y
#    
#    def __len__(self):
#        return math.ceil(len(self.dataset) / self.batch_size)