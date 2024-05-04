import numpy as np
import ntimage as nti

from ..loaders import Loader

class Predictor:
    
    def __init__(self, model, task, sampler=None, expand_dims=-1):
        self.model = model
        self.task = task
        self.sampler = sampler
        self.expand_dims = expand_dims
    
    def predict(self, dataset):
        """
        Perform inference on an nt.Dataset.
        
        The function performs inference on the supplied object using
        the fitted model intialized with the predictor. The object
        will go through the sampler before inference but will maintain
        its original shape.
        
        The task determines whether the resulting inference is converted
        to an image or not and whether the prediction values are rounded
        to be in the same style as the dataset output.
        
        The result of the prediction will be one (or a sequence of) of the 
        following depending on the model: ntimage, np.ndarray, scalar.
        """
        
        y_pred_list = []
        for x, y in dataset:
            sampled_batch = self.sampler([x], [y])
            
            y_pred = []
            for x_batch, y_batch in sampled_batch:
                if isinstance(x_batch[0], list):
                    x_batch_return = []
                    for i in range(len(x_batch[0])):
                        tmp_x_batch = np.array([np.expand_dims(xx[i].numpy(), self.expand_dims) if self.expand_dims else xx.numpy() for xx in x_batch])
                        x_batch_return.append(tmp_x_batch)
                    x_batch = x_batch_return
                else:
                    x_batch = np.array([np.expand_dims(xx.numpy(), self.expand_dims) if self.expand_dims else xx.numpy() for xx in x_batch])
                
                # TODO: write general function for model prediction
                tmp_y_pred = self.model.predict(x_batch)
                y_pred.append(tmp_y_pred)
            
            # TODO: handle multiple inputs
            y_pred = np.concatenate(y_pred)
            y_pred = np.squeeze(y_pred)
            
            # put sampled axis in correct place
            if 'SliceSampler' in str(type(self.sampler)):
                y_pred = np.rollaxis(y_pred, 0, self.sampler.axis) 
            
            # process prediction according to task
            if y_pred.ndim > 1:
                if self.task == 'segmentation' or self.task == 'classification':
                    y_pred = np.round(y_pred).astype('uint8')
                y_pred = nti.from_numpy(y_pred)
                
            y_pred_list.append(y_pred)

        return y_pred_list

            
        
        