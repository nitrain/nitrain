import ntimage as nti

import random
import numpy as np

from .base import BaseTransform

class Astype(BaseTransform):
    """
    Cast an image to another datatype
    """
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, *images):
        images = [image.astype(self.dtype) for image in images]
        return images if len(images) > 1 else images[0]

class StandardNormalize(BaseTransform):
    """
    import ntimage as nt
    img = nti.load(nti.example_data('r16'))
    img2 = (img - img.mean()) / img.std()
    
    from nitrain import transforms as tx
    my_tx = tx.StandardNormalize()
    img3 = my_tx(img)
    """
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std
        
    def __call__(self, *images):
        new_images = []
        for img in images:
            mean_val = self.mean if self.mean else img.mean()
            std_val = self.std if self.std else img.std()
            new_image = (img - mean_val) / std_val
            new_images.append(new_image)
        
        return new_images if len(new_images) > 1 else new_images[0]


class Threshold(BaseTransform):
    """
    import ntimage as nt
    img = nti.load(nti.example_data('r16'))
    """
    def __init__(self, threshold):
        self.threshold = threshold
        
    def __call__(self, *images):
        new_images = []
        for image in images:
            new_image = image * (image > self.threshold)
            new_images.append(new_image)
        return new_images if len(new_images) > 1 else new_images[0]


class RangeNormalize(BaseTransform):
    def __init__(self, min=0, max=1, level='individual'):
        self.min = min
        self.max = max
        
    def __call__(self, *images):
        new_images = []
        for image in images:
            min_val = self.min if self.min else image.min()
            max_val = self.max if self.max else image.max()
            new_image = (image - min_val) / (max_val - min_val)
            new_images.append(new_image)
        return new_images if len(new_images) > 1 else new_images[0]


class Smooth(BaseTransform):
    """
    import ntimage as nt
    img = nti.load(nti.example_data('r16'))
    img2 = nti.smooth(img, 2, True)
    img3 = nti.smooth(img, 2, False)
    """
    
    def __init__(self, sigma):
        """
        Arguments
        ---------
        sigma : float or tuple of floats
            std of a Gaussian kernal.
        """
        self.sigma = sigma
        
    def __call__(self, *images):
        new_images = []
        for image in images:
            new_image = nti.smooth(image, self.sigma, method='gaussian')
            new_images.append(new_image)
        return new_images if len(new_images) > 1 else new_images[0]