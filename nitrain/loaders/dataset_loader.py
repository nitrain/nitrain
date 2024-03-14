import math
import numpy as np

from .. import samplers

class DatasetLoader:
    
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
        
    def as_keras_loader(self):
        def my_func(batch_size, dataset, x_transforms, y_transforms, co_transforms,
                    sampler, expand_dims):
            n_batches = math.ceil(len(dataset) / batch_size)
            
            # TODO: apply shuffling to indices before we get to the batch loop
            
            batch_idx = 0
            while True:
                if batch_idx >= n_batches:
                    batch_idx = 0
                
                data_indices = slice(batch_idx*batch_size, min((batch_idx+1)*batch_size, len(dataset)))
                x, y = dataset[data_indices]
                
                batch_idx += 1
                # perform transforms
                if self.x_transforms:
                    for tx_fn in x_transforms:
                        x = [tx_fn(xx) for xx in x]
                
                if self.y_transforms:
                    for tx_fn in y_transforms:
                        y = [tx_fn(yy) for yy in y]
                
                if self.co_transforms:
                    for tx_fn in co_transforms:
                        for i in range(len(x)):
                            x[i], y[i] = tx_fn(x[i], y[i])

                # sample the batch
                sampled_batch = sampler(x, y)
                
                # a normal sampler will just return the entire (shuffled, if specified) batch once
                # a slice sampler will return shuffled slices with batch size = sampler.batch_size
                for x_batch, y_batch in sampled_batch:
                    if self.expand_dims is not None:
                        x_batch = np.array([np.expand_dims(xx.numpy(), expand_dims) for xx in x_batch])
                    else:
                        x_batch = np.array([xx.numpy() for xx in x_batch])
                    
                    yield x_batch, y_batch

        return my_func(self.batch_size, self.dataset, self.x_transforms, self.y_transforms, self.co_transforms,
                    self.sampler, self.expand_dims)
                
    def __iter__(self):
        batch_size = self.batch_size
        dataset = self.dataset
        n_batches = math.ceil(len(dataset) / batch_size)
        
        # TODO: apply shuffling to indices before we get to the batch loop
        
        batch_idx = 0
        while batch_idx < n_batches:
            
            batch_idx += 1
            
            data_indices = slice(batch_idx*batch_size, min((batch_idx+1)*batch_size, len(dataset)))
            x, y = dataset[data_indices]
           
            # perform transforms
            if self.x_transforms:
                for tx_fn in self.x_transforms:
                    x = [tx_fn(xx) for xx in x]
            
            if self.y_transforms:
                for tx_fn in self.y_transforms:
                    y = [tx_fn(yy) for yy in y]
            
            if self.co_transforms:
                for tx_fn in self.co_transforms:
                    for i in range(len(x)):
                        x[i], y[i] = tx_fn(x[i], y[i])

            # sample the batch
            sampled_batch = self.sampler(x, y)
            
            # a normal sampler will just return the entire (shuffled, if specified) batch once
            # a slice sampler will return shuffled slices with batch size = sampler.batch_size
            for x_batch, y_batch in sampled_batch:
                
                if self.expand_dims is not None:
                    x_batch = np.array([np.expand_dims(xx.numpy(), self.expand_dims) for xx in x_batch])
                else:
                    x_batch = np.array([xx.numpy() for xx in x_batch])
                
                yield x_batch, y_batch
                
            

    def __len__(self):
        return math.ceil((len(self.dataset) / self.batch_size) * len(self.sampler))
    

