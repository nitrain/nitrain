from .base import BaseTransform

__all__ = [
    'Apply',
    'Affine',
    'Shear',
    'Rotate',
    'Zoom',
    'Flip',
    'Translate'
]

class Apply(BaseTransform):
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, *images):
        images = [image.apply(self.transform) for image in images]
        return images if len(images) > 1 else images[0]

class Affine(BaseTransform):
    def __init__(self, array):
        self.array = array
        
    def __call__(self, *images):
        images = [image.affine(self.array) for image in images]
        return images if len(images) > 1 else images[0]

class Shear(BaseTransform):
    def __init__(self, shear, axis=0):
        self.shear = shear
        self.axis = axis
        
    def __call__(self, *images):
        images = [image.shear(self.shear, self.axis) for image in images]
        return images if len(images) > 1 else images[0]

class Rotate(BaseTransform):
    def __init__(self, theta, axis=0):
        self.theta = theta
        self.axis = axis
        
    def __call__(self, *images):
        images = [image.rotate(self.theta, self.axis) for image in images]
        return images if len(images) > 1 else images[0]

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
    
class Translate(BaseTransform):
    def __init__(self, translation):
        self.translation = translation
        
    def __call__(self, *images):
        images = [image.translate(self.translation) for image in images]
        return images if len(images) > 1 else images[0]