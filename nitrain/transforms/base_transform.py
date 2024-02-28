# transforms perform some function that alters your images
import os
import numpy as np

class BaseTransform:
    
    def fit(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        raise NotImplementedError








