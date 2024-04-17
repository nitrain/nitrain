import os
import ntimage as nt
import random
import string
import numpy as np

from .base_transform import BaseTransform

class ToFile(BaseTransform):
    """
    Saves an image to file using `nt.plot`. 
    
    Useful as a pass-through ransform when wanting to observe 
    how augmentation affects the data.
    """
    def __init__(self, base_dir, ortho=False):
        self.base_dir = base_dir
        self.ortho = ortho

    def __call__(self, image):
        filename = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
        if self.ortho:
            nt.plot_ortho(image, filename=os.path.join(self.base_dir, filename+'.png'))
        else:
            nt.plot(image, filename=os.path.join(self.base_dir, filename+'.png'))
        return image


class CustomFunction(BaseTransform):
    """
    Apply a user-supplied function as a transform
    """
    
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, image):
        new_image = self.fn(image)
        return new_image


class RandomChoice(BaseTransform):
    """
    Randomly choose which of the supplied transforms to apply
    based on the supplied probabilities
    """
    
    def __init__(self, *transforms, p=None):
        self.transforms = transforms
        if p is None:
            p = [1.0 / len(self.transforms) for _ in len(self.transforms)]
        self.p = p
    
    def __call__(self, image):
        chosen_tx = np.random.choice(self.transforms, size = 1, p = self.p)[0]
        new_image = chosen_tx(image)
        return new_image