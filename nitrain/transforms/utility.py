import os
import ntimage as nt
import random
import string
import numpy as np

from .base import BaseTransform

class Custom(BaseTransform):
    """
    Apply a user-supplied function as a transform
    """
    
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *images):
        images = [self.fn(image) for image in images]
        return images if len(images) > 1 else images[0]
