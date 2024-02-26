# loaders help you feed your data into deep learning models with transforms
import math
import numpy as np
import torch

class TransformDataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset, x_transforms=None):
        self.dataset = dataset
        self.x_transforms = x_transforms
        
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.x_transforms is not None:
            x = self.x_transforms(x)
            
        return x, y

    def __len__(self):
        return len(self.dataset)
    

class DatasetLoader(torch.utils.data.DataLoader):
    
    def __init__(self, dataset, batch_size, x_transforms=None, y_transforms=None, co_transforms=None):
        transform_dataset = TransformDataset(dataset,
                                             x_transforms=x_transforms)
        super(DatasetLoader, self).__init__(transform_dataset, batch_size=batch_size)

class DatasetLoader2:
    
    def __init__(self, dataset, batch_size, x_transforms=None, y_transforms=None, co_transforms=None):
        self.dataset = dataset
        self.batch_size = batch_size
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x = np.expand_dims(x.numpy(), -1)
        return x, y
    
    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            
    #def __call__(self):
    #    batch_idx = 0
    #    batch_size = self.batch_size
    #    dataset = self.dataset
    #    while batch_idx < self.n_batches():
    #        data_indices = slice(batch_idx*batch_size, min((batch_idx+1)*batch_size, len(dataset)))
    #        imgs, y = self.dataset[data_indices]
    #        x = np.array([img.numpy() for img in imgs], dtype='float32')
    #        x = np.expand_dims(x, -1)
    #        yield x, y
    #        batch_idx += 1
    
    def __iter__(self):
       batch_idx = 0
       batch_size = self.batch_size
       dataset = self.dataset
       
       while batch_idx < self.n_batches():
           data_indices = slice(batch_idx*batch_size, min((batch_idx+1)*batch_size, len(dataset)))
           imgs, y = self.dataset[data_indices]
           x = np.array([img.numpy() for img in imgs], dtype='float32')
           x = np.expand_dims(x, -1)
           yield x, y
           batch_idx += 1
    
    def n_batches(self):
        return math.ceil(len(self.dataset) / self.batch_size)
    
    @property
    def input_shape(self):
        # load the first input, perform any transforms, and get shape
        return self.dataset[0][0].shape

class Counter:
    def __init__(self, start, end):
        self.num = start
        self.end = end
 
    def __iter__(self):
        return self
 
    def __next__(self): 
        if self.num > self.end:
            raise StopIteration
        else:
            self.num += 1
            return self.num - 1