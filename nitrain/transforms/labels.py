from .base import BaseTransform

__all__ = [
    'LabelsToChannels'
]

class LabelsToChannels(BaseTransform):
    """
    Create a channel dimension for each separate value in a
    segmentation image.
    
    If an image has shape (100,100) and has three unique values (0,1,2),
    then this transform will return an image with shape (100,100,3) where
    (100,100,0) = 1 if the original value is 0, (100,100,1) = 1 if the original
    value is 1, and (100,100,2) = 2 if the original value is 2. 
    
    It is also possible to keep the original values in the channels 
    instead of making all values equal to 1.
    """
    def __init__(self, keep_values=False):
        self.keep_values = keep_values
    
    def __call__(self, *images):
        images = [image.labels_to_channels(self.keep_values) for image in images]
        return images if len(images) > 1 else images[0]