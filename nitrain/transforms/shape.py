from .base import BaseTransform

__all__ = [
    'Reorient',
    'Rollaxis',
    'Repeat'
]


class Reorient(BaseTransform):
    def __init__(self, orientation):
        self.orientation = orientation
        
    def __call__(self, *images):
        images = [image.reorient(self.orientation) for image in images]
        return images if len(images) > 1 else images[0]


class Rollaxis(BaseTransform):
    def __init__(self, axis, start=0):
        self.axis = axis
        self.start = start
        
    def __call__(self, *images):
        images = [image.rollaxis(self.axis, self.start) for image in images]
        return images if len(images) > 1 else images[0]
    
class Repeat(BaseTransform):
    def __init__(self, n):
        self.n = n
    def __call__(self, *images):
        images = [image.repeat(self.n) for image in images]
        return images if len(images) > 1 else images[0]
    