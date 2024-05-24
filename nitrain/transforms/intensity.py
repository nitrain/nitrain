from .base import BaseTransform

__all__ = ['ImageMath',
           'BiasCorrection',
           'StandardNormalize',
           'RangeNormalize',
           'Clip',
           'QuantileClip',
           'Threshold']

class ImageMath(BaseTransform):
    def __init__(self, operation, *args):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.ImageMath('Canny', 1, 5, 12)
        img2 = mytx(img)
        """
        self.operation = operation
        self.args = args
    
    def __call__(self, *images):
        images = [image.iMath(self.operation, *self.args) for image in images]
        return images if len(images) > 1 else images[0]
    
class BiasCorrection(BaseTransform):
    def __init__(self):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.BiasCorrection()
        img2 = mytx(img)
        """
        pass

    def __call__(self, *images):
        images = [image.n4_bias_field_correction() for image in images]
        return images if len(images) > 1 else images[0]

class StandardNormalize(BaseTransform):
    def __init__(self, mean=None, std=None):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.StandardNormalize()
        img2 = mytx(img)
        """
        self.mean = mean
        self.std = std

    def __call__(self, *images):
        new_images = []
        for image in images:
            image = image.clone('float')
            mean_val = self.mean if self.mean else image.mean()
            std_val = self.std if self.std else image.std()
            image = (image - mean_val) / std_val
            new_images.append(image)
        return new_images if len(new_images) > 1 else new_images[0]
    
class RangeNormalize(BaseTransform):
    def __init__(self, min=0, max=1):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.RangeNormalize(0, 1)
        img2 = mytx(img)
        """
        self.min = min
        self.max = max
    
    def __call__(self, *images):
        new_images = []
        for image in images:
            image = image.clone('float')
            minimum = image.min()
            maximum = image.max()
            if maximum - minimum == 0:
                new_images.append(image)
            else:
                m = (self.max - self.min) / (maximum - minimum)
                b = self.min - m * minimum
                image = m * image + b
                new_images.append(image)
        return new_images if len(new_images) > 1 else new_images[0]
    
class Clip(BaseTransform):
    def __init__(self, lower, upper):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.Clip(10, 200)
        img2 = mytx(img)
        """
        self.lower = lower
        self.upper = upper
    
    def __call__(self, *images):
        new_images = []
        for image in images:
            image[image < self.lower] = self.lower
            image[image > self.upper] = self.upper
            new_images.append(image)

        return new_images if len(new_images) > 1 else new_images[0]


class QuantileClip(BaseTransform):
    def __init__(self, lower, upper):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.QuantileClip(0.1, 0.9)
        img2 = mytx(img)
        """
        self.lower = lower
        self.upper = upper
    
    def __call__(self, *images):
        new_images = [image.iMath_truncate_intensity(self.lower, self.upper) for image in images]
        return new_images if len(new_images) > 1 else new_images[0]
    

class Threshold(BaseTransform):
    def __init__(self, value, as_upper=False):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mymytx = tx.Threshold(10)
        img2 = mytx(img)
        mymytx = tx.Threshold(200, as_upper=True)
        img3 = mytx(img)
        """
        self.value = value
        self.as_upper = as_upper
    
    def __call__(self, *images):
        if self.as_upper:
            images = [image.threshold_image(high_thresh=self.value, binary=False) for image in images]
        else:
            images = [image.threshold_image(low_thresh=self.value, binary=False) for image in images]
        return images if len(images) > 1 else images[0]