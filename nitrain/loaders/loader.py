import math
import numpy as np
import warnings
import ntimage as nti
from copy import deepcopy

from .. import samplers, transforms as tx
from ..datasets.utils import reduce_to_list, apply_transforms

class Loader:
    
    def __init__(self,
                 dataset, 
                 images_per_batch, 
                 transforms=None,
                 expand_dims=-1,
                 shuffle=False,
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
        if images_per_batch > len(dataset):
            warnings.warn(f'Warning: The supplied images_per_batch ({images_per_batch}) is larger than available dataset records ({len(dataset)}). Setting to available dataset records.')
            images_per_batch = len(dataset)
            
        self.dataset = dataset
        self.images_per_batch = images_per_batch
        self.expand_dims = expand_dims
        self.transforms = transforms
        self.shuffle = shuffle
        
        if sampler is None:
            sampler = samplers.BaseSampler(batch_size=images_per_batch)
        self.sampler = sampler
        
    def copy(self, dataset=None):
        new_loader = deepcopy(self)
        new_loader.dataset = dataset
        return new_loader
        
    def to_keras(self, output_signature=None):
        import tensorflow as tf
 
        # generate a training batch to infer the output signature
        if output_signature is None:
            tmp_batch_size = self.images_per_batch
            self.images_per_batch = 1
            x_batch, y_batch = next(iter(self))
            self.images_per_batch = tmp_batch_size
            if isinstance(x_batch, list):
                x_spec = tuple([tf.type_spec_from_value(xb[0]) for xb in x_batch])
            else:
                x_spec = tf.type_spec_from_value(x_batch[0])
            if isinstance(y_batch, list) and nti.is_image(y_batch[0]):
                y_spec = tuple([tf.type_spec_from_value(yb[0]) for yb in y_batch])
            else:
                y_spec = tf.type_spec_from_value(y_batch[0])
        
        generator = tf.data.Dataset.from_generator(
            lambda: record_generator(self),
            output_signature=(x_spec, y_spec)
        ).batch(self.sampler.batch_size)
        
        return generator
                
    def __iter__(self):
        images_per_batch = self.images_per_batch
        dataset = self.dataset
        n_image_batches = math.ceil(len(dataset) / images_per_batch)
        
        original_indices = np.arange(len(dataset))
        if self.shuffle:
            np.random.shuffle(original_indices)
        
        image_batch_idx = 0
        while image_batch_idx < n_image_batches:
            
            # TODO: implement shuffle here 
            data_indices = slice(image_batch_idx*images_per_batch, min((image_batch_idx+1)*images_per_batch, len(dataset)))
            x, y = dataset[data_indices, self.transforms is None]

            if self.transforms:
                x, y = transform_records(x, y, self.transforms)

            image_batch_idx += 1
            
            # sample the batch
            sampled_batch = self.sampler(x, y)
            
            for x_batch, y_batch in sampled_batch:

                if self.expand_dims:
                    x_batch = expand_image_dims(x_batch)
                    y_batch = expand_image_dims(y_batch)
                
                x_batch = convert_to_numpy(x_batch)
                y_batch = convert_to_numpy(y_batch)
                
                yield x_batch, y_batch
                
    def __len__(self):
        # TODO: take into account batch_size from sampler ?
        # issue: requires loading at least one record from dataset
        # issue: nslices, for example, may not be the same for all images in dataset
        return math.ceil(len(self.dataset) / self.images_per_batch)
    
    def __repr__(self):
        s = 'Loader (batches={})\n'.format(self.__len__())
        
        s = s +\
            '   {}\n'.format(repr(self.dataset))+\
            '   {} : {}\n'.format('Transforms', len(self.transforms) if self.transforms else '{}')
        return s


def transform_records(x_list, y_list, transforms):
    x_items = []
    y_items = []
    for x, y in zip(x_list, y_list):
        for tx_name, tx_value in transforms.items():
            x, y = apply_transforms(tx_name, tx_value, x, y)
                                
        x = reduce_to_list(x)
        y = reduce_to_list(y)
        
        x_items.append(x)
        y_items.append(y)
    
    return x_items, y_items
    
def convert_to_numpy(x):
    """
    img = nti.example('r16')
    x = [[img,img,img], [img, img, img]]
    x2 = convert_to_numpy(x)
    """
    if isinstance(x[0], list):
        return [convert_to_numpy(xx) for xx in x]
    if nti.is_image(x[0]):
        return np.array([xx.numpy() for xx in x])
    else:
        return np.array(x)

def expand_image_dims(x):
    mytx = tx.ExpandDims()
    if isinstance(x, list):
        return [expand_image_dims(xx) for xx in x]
    else:
        return mytx(x) if nti.is_image(x) and not nti.has_channels(x) else x
    
def record_generator(loader):
    """
    This function takes a batch and returns individual records from it.
    
    It is necessary for tf.keras generators
    """
    my_iter = iter(loader)
    for x_batch, y_batch in my_iter:
        if isinstance(x_batch, list) and isinstance(x_batch[0], np.ndarray):
            if isinstance(y_batch, list) and isinstance(y_batch[0], np.ndarray):
                for i in range(len(x_batch[0])):
                    yield tuple([xb[i] for xb in x_batch]), tuple([yb[i] for yb in y_batch])
            else:
                for i in range(len(x_batch[0])):
                    yield tuple([xb[i] for xb in x_batch]), y_batch[i]
        else:
            if isinstance(y_batch, list) and isinstance(y_batch[0], np.ndarray):
                for i in range(len(x_batch)):
                    yield x_batch[i], tuple([yb[i] for yb in y_batch])
            else:
                for i in range(len(x_batch)):
                    yield x_batch[i], y_batch[i]