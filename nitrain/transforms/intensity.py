from .base import BaseTransform

__all__ = ['StdNormalize',
           'Normalize',
           'Clamp',
           'Threshold']

class StdNormalize(BaseTransform):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, *images):
        images = [image.std_normalize(self.mean, self.std) for image in images]
        return images if len(images) > 1 else images[0]
    
class Normalize(BaseTransform):
    def __init__(self, min, max):
        self.min = min
        self.max = max
    
    def __call__(self, *images):
        images = [image.normalize(self.min, self.max) for image in images]
        return images if len(images) > 1 else images[0]
    
class Clamp(BaseTransform):
    def __init__(self, min, max):
        self.min = min
        self.max = max
    
    def __call__(self, *images):
        images = [image.clamp(self.min, self.max) for image in images]
        return images if len(images) > 1 else images[0]
    
    
class Threshold(BaseTransform):
    def __init__(self, value, use_below=False):
        self.value = value
        self.use_below = use_below
    
    def __call__(self, *images):
        images = [image.threshold(self.value, self.use_below) for image in images]
        return images if len(images) > 1 else images[0]