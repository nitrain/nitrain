import ants

import random
import numpy as np

from .base_transform import BaseTransform


class StandardNormalize(BaseTransform):
    """
    import ants
    img = ants.image_read(ants.get_ants_data('r16'))
    img2 = (img - img.mean()) / img.std()
    
    from nitrain import transforms as tx
    my_tx = tx.StandardNormalize()
    img3 = my_tx(img)
    """
    def __init__(self, level='individual'):
        self.level = level
        
    def __call__(self, *images):
        new_images = []
        for img in images:
            new_image = (img - img.mean()) / img.std()
            new_images.append(new_image)
        
        return new_images if len(new_images) > 1 else new_images[0]


class Threshold(BaseTransform):
    """
    import ants
    img = ants.image_read(ants.get_ants_data('r16'))
    """
    def __init__(self, threshold):
        self.threshold = threshold
        
    def __call__(self, *images):
        new_images = []
        for image in images:
            new_image = image * (image > self.threshold)
            new_images.append(new_image)
        return new_images if len(new_images) > 1 else new_images[0]


class RangeNormalize(BaseTransform):
    def __init__(self, min=0, max=1, level='individual'):
        self.min = min
        self.max = max
        
    def __call__(self, *images):
        new_images = []
        for image in images:
            new_image = (image - image.min()) / (image.max() - image.min())
            new_images.append(new_image)
        return new_images if len(new_images) > 1 else new_images[0]


class Smoothing(BaseTransform):
    """
    import ants
    img = ants.image_read(ants.get_ants_data('r16'))
    img2 = ants.smooth_image(img, 2, True)
    img3 = ants.smooth_image(img, 2, False)
    """
    
    def __init__(self, std, physical_space=True):
        """
        Arguments
        ---------
        std : float or tuple of floats
            std of a Gaussian kernal.

        physical_space : boolean
            If true, std is interpreted as being in millimeters (i.e., physical coordinates).
            If false, std is interpreted as being in pixels. This makes no difference if 
            the image has unit spacing.
        """
        self.std = std
        self.physical_space = physical_space
        
    def __call__(self, *images):
        new_images = []
        for image in images:
            new_image = ants.smooth_image(image, 
                                          self.std,
                                          physical_space=self.physical_space)
            new_images.append(new_image)
        return new_images if len(new_images) > 1 else new_images[0]


class RandomSmoothing(BaseTransform):

    def __init__(self, min_std, max_std, physical_space=True):
        self.min_std = min_std
        self.max_std = max_std
        self.physical_space = physical_space

    def __call__(self, *images):
        std = random.uniform(self.min_std, self.max_std)
        
        new_images = []
        for image in images:
            new_image = ants.smooth_image(image, 
                                          std,
                                          self.physical_space)
            new_images.append(new_image)
        return new_images if len(new_images) > 1 else new_images[0]
        

class RandomNoise(BaseTransform):
    """
    Apply random additive gaussian noise to an image.
    
    import ants
    img = ants.image_read(ants.get_ants_data('r16'))
    img2 = ants.add_noise_to_image(img, 'additivegaussian', (0, 16))
    ants.plot((img2 - img))
    """
    def __init__(self, min_std, max_std):
        self.min_std = min_std
        self.max_std = max_std
    
    def __call__(self, *images):
        std = random.uniform(self.min_std, self.max_std)
        new_images = []
        for image in images:
            new_image = ants.add_noise_to_image(image,
                                                'additivegaussian',
                                                (0, std))
            new_images.append(new_image)
        return new_images if len(new_images) > 1 else new_images[0]


class HistogramWarpIntensity(BaseTransform):
    """
    See https://github.com/ANTsX/ANTsPyNet/blob/master/antspynet/utilities/histogram_warp_image_intensities.py
    """
    def __init__(self, 
                 break_points=(0.25, 0.5, 0.75),
                 displacements=None,
                 clamp_end_points=(False, False),
                 sd_displacements=0.05,
                 transform_domain_size=20):
        self.break_points = break_points
        self.displacements = displacements
        self.clamp_end_points = clamp_end_points
        self.sd_displacements = sd_displacements
        self.transform_domain_size = transform_domain_size

    def __call__(self, *images):
        import antspynet
        
        new_images = []
        for image in images:
            new_image = antspynet.histogram_warp_image_intensities(
                image = image,
                break_points = self.break_points,
                displacements = self.displacements,
                clamp_end_points = self.clamp_end_points,
                sd_displacements = self.sd_displacements,
                transform_domain_size = self.transform_domain_size
            )
            new_images.append(new_image)
        return new_images if len(new_images) > 1 else new_images[0]