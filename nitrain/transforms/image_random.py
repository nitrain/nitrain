
import numpy as np

from .base import BaseTransform

__all__ = [
    'RandomCrop'
]

class RandomCrop(BaseTransform):
    def __init__(self, shape):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.RandomCrop((98,98))
        img2 = mytx(img)
        """
        self.shape = shape
        
    def __call__(self, *images):
        # TODO: support aligned crops on images of different size via registration
        image_shape = images[0].shape
        ndim = len(image_shape)
        indices = [np.random.choice(np.arange(0, image_shape[0]-self.shape[i]+1, step=1)) 
                   for i in range(ndim)]
        images = [image.crop_indices([indices[i] for i in range(ndim)], 
                                     [indices[i]+self.shape[i] for i in range(ndim)]) 
                  for image in images]
        return images if len(images) > 1 else images[0]

