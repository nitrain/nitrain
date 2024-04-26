# transforms perform some function that alters your images
import os
import numpy as np

class BaseTransform:
    
    def __init__(self, prob=1):
        self.prob = prob
        
    def fit(self, *inputs):
        raise NotImplementedError

    def __call__(self, *inputs):
        raise NotImplementedError

    def sample(self, image, n=10):
        res = []
        for _ in range(n):
            res.append(self.__call__(image))
        return res
        
    def __call__(self, *inputs):
        raise NotImplementedError

    def __repr__(self):
        raise NotImplementedError








