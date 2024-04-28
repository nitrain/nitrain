from .base import BaseTransform

__all__ = [
    'Zoom',
    'Flip'
]


class Zoom(BaseTransform):
    def __init__(self, scale):
        self.scale = scale
        
    def __call__(self, *images):
        images = [image.zoom(self.scale) for image in images]
        return images if len(images) > 1 else images[0]

class Flip(BaseTransform):
    def __init__(self, axis):
        self.axis = axis
        
    def __call__(self, *images):
        images = [image.flip(self.axis) for image in images]
        return images if len(images) > 1 else images[0]