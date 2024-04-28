from .base import BaseTransform

__all__ = [
    'Zoom'
]


class Zoom(BaseTransform):
    def __init__(self, scale):
        self.scale = scale
        
    def __call__(self, *images):
        images = [image.zoom(self.scale) for image in images]
        return images if len(images) > 1 else images[0]