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
            sampler = samplers.BaseSampler(sub_batch_size=batch_size)
        self.sampler = sampler
        
    def to_keras(self, output_signature=None):
        import tensorflow as tf
        
        def batch_generator():
            my_iter = iter(self)
            for x_batch, y_batch in my_iter:
                for i in range(x_batch.shape[0]):
                    yield x_batch[i,:], y_batch[i]
 
        # generate a training batch to infer the output signature
        if output_signature is None:
            tmp_batch_size = self.batch_size
            self.batch_size = 1
            x_batch, y_batch = next(iter(self))
            self.batch_size = tmp_batch_size
            x_spec = tf.type_spec_from_value(x_batch[0,:])
            y_spec = tf.type_spec_from_value(y_batch[0])
        
        generator = tf.data.Dataset.from_generator(
            lambda: batch_generator(),
            output_signature=(x_spec, y_spec)
        ).batch(self.sampler.sub_batch_size)
        
        return generator
                
    def __iter__(self):
        batch_size = self.batch_size
        dataset = self.dataset
        n_image_batches = math.ceil(len(dataset) / batch_size)
        
        # TODO: shuffle image indices before batch loop
        
        image_batch_idx = 0
        while image_batch_idx < n_image_batches:
            
            data_indices = slice(image_batch_idx*batch_size, min((image_batch_idx+1)*batch_size, len(dataset)))
            x, y = dataset[data_indices]
            
            image_batch_idx += 1
           
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
    

