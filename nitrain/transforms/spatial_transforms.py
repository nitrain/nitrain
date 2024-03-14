import ants
import numpy as np
import random

from .base_transform import BaseTransform

class RandomAffine(BaseTransform):
    pass

class RandomRotate(BaseTransform):
    pass

class RandomTranslate(BaseTransform):
    pass

class RandomShear(BaseTransform):
    pass

class RandomZoom(BaseTransform):
    """
    Apply a random zoom transform to an image
    
    Examples
    --------
    >>> import ants
    >>> from nitrain import transforms as tx
    >>> image = ants.image_read(ants.get_data('r16'))
    >>> my_tx = tx.RandomZoom(0.8,1.2)
    >>> new_image = my_tx(image)
    """
    
    def __init__(self, min_zoom, max_zoom):
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
    
    def __call__(self, image):
        zoom = random.uniform(self.min_zoom, self.max_zoom)
        if image.dimension == 2:
            matrix =  np.array([[zoom, 0], [0, zoom]])
        else:
            matrix =  np.array([[zoom, 0, 0], [0, zoom, 0], [0, 0, zoom]])
        transform = create_centered_affine_transform(image, matrix)
        new_image = transform.apply_to_image(image)
        return new_image
        

class RandomFlip(BaseTransform):
    """
    Randomly flip an image with specified proabilikty.
    
    If no axis is specified, then the axis will be chosen
    randomly with equal probability.
    
    Examples
    --------
    >>> import ants
    >>> from nitrain import transforms as tx
    >>> img = ants.image_read(ants.get_data('r16'))
    >>> my_tx = tx.RandomFlip(p=1)
    >>> new_img = my_tx(img)
    >>> new_img.plot()
    """
    def __init__(self, p=0.5, axis=None):
        self.p = p
        self.axis = axis
        
    def __call__(self, image):
        apply_tx = np.random.choice([True, False], size=1, p=[self.p, 1-self.p])[0]
        
        axis = self.axis
        if axis is None:
            axis = np.random.choice(range(image.dimension), size=1)[0]
            
        if apply_tx:
            image = image.reflect_image(axis=axis)
            
        return image
        

def create_centered_affine_transform(image, matrix):
    transform = ants.create_ants_transform(
        transform_type="AffineTransform", 
        precision='float', 
        matrix=matrix,
        center=[image.shape[i]/2 for i in range(image.dimension)],
        dimension=image.dimension
    )
    return transform