
import math
import random
import numpy as np
import ants

from .base import BaseTransform
from .spatial import Shear, Rotate, Zoom, Flip, Translate

__all__ = [
    'RandomShear',
    'RandomRotate',
    'RandomZoom',
    'RandomFlip',
    'RandomTranslate',
]

class RandomShear(BaseTransform):
    def __init__(self, min_shear, max_shear, reference=None, p=1):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.RandomShear((0,-10), (0,10))
        img2 = mytx(img)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.RandomShear((0,-10,0),(0,10,0))
        img2 = mytx(img)
        """
        if not isinstance(min_shear, (list, tuple)) or not isinstance(max_shear, (list, tuple)):
            raise Exception('The min_shear and max_shear args must be list or tuple with length equal to image dimension.')
        
        self.max_shear = min_shear
        self.min_shear = max_shear
        self.reference = reference
        self.p = p
        
    def __call__(self, *images):
        if random.uniform(0, 1) > self.p:
            return images if len(images) > 1 else images[0]
        
        shear = [random.uniform(min_s, max_s) for min_s, max_s in zip(self.min_shear, self.max_shear)]
        mytx = Shear(shear, self.reference)
        new_images = [mytx(image) for image in images]
        return new_images if len(new_images) > 1 else new_images[0]

class RandomRotate(BaseTransform):
    def __init__(self, min_rotation, max_rotation, reference=None, p=1):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.RandomRotate(-10, 10)
        img2 = mytx(img)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.RandomRotate((-90,0,0), (90,0,0), img)
        img2 = mytx(img)
        """
        self.min_rotation = min_rotation
        self.max_rotation = max_rotation
        self.reference = reference
        self.p = p
        
    def __call__(self, *images):        
        if random.uniform(0, 1) > self.p:
            return images if len(images) > 1 else images[0]
        
        if isinstance(self.min_rotation, (tuple, list)):
            rotation = [random.uniform(min_r, max_r) for min_r, max_r in zip(self.min_rotation, self.max_rotation)]
        else:
            rotation = random.uniform(self.min_rotation, self.max_rotation)
        
        mytx = Rotate(rotation, self.reference)
        new_images = [mytx(image) for image in images]
        return new_images if len(new_images) > 1 else new_images[0]
    
    def __repr__(self):
        return 'RandomRotation'
        

class RandomZoom(BaseTransform):
    def __init__(self, min_zoom, max_zoom, reference=None, p=1):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.RandomZoom(0.9, 1.1)
        img2 = mytx(img)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.RandomZoom(0.9, 1.1)
        img2 = mytx(img)
        """
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.reference = reference
        self.p = p

    def __call__(self, *images):
        if random.uniform(0, 1) > self.p:
            return images if len(images) > 1 else images[0]
        
        zoom = random.uniform(self.min_zoom, self.max_zoom)
        mytx = Zoom(zoom, self.reference)
        new_images = [mytx(image) for image in images]
        return new_images if len(new_images) > 1 else new_images[0]


class RandomFlip(BaseTransform):
    
    def __init__(self, axis=0, p=0.5):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.RandomFlip()
        img2 = mytx(img)
        """
        self.axis = axis
        self.p = p
        
    def __call__(self, *images):
        if random.uniform(0, 1) > self.p:
            return images if len(images) > 1 else images[0]
        
        mytx = Flip(self.axis)
        new_images = [mytx(image) for image in images]
        return new_images if len(new_images) > 1 else new_images[0]
    
class RandomTranslate(object):
    def __init__(self, min_translation, max_translation, reference=None, p=1):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.RandomTranslate((-10,-30), (10, 30))
        img2 = mytx(img)
        
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.Translate((-30,-30,-20), (30,30,20))
        img2 = mytx(img)
        """
        if not isinstance(min_translation, (list, tuple)) or not isinstance(max_translation, (list, tuple)):
            raise Exception("The min_translation and max_translation arguments must be list or tuple.")

        self.min_translation = min_translation
        self.max_translation = max_translation
        self.reference = reference
        self.p = p

    def __call__(self, *images):
        if random.uniform(0, 1) > self.p:
            return images if len(images) > 1 else images[0]
        
        translation = [random.uniform(min_t, max_t) for min_t, max_t in zip(self.min_translation, self.max_translation)]
        mytx = Translate(translation, self.reference)
        new_images = [mytx(image) for image in images]
        return new_images if len(new_images) > 1 else new_images[0]