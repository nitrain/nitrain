
from .base import BaseTransform

__all__ = [
    'Astype',
    'Smooth',
    'Crop',
    'Resample',
    'Slice',
    'Pad'
]

class Astype(BaseTransform):
    def __init__(self, dtype):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.Astype('float32')
        img2 = mytx(img)
        """
        self.dtype = dtype
    def __call__(self, *images):
        images = [image.astype(self.dtype) for image in images]
        return images if len(images) > 1 else images[0]

class Smooth(BaseTransform):
    def __init__(self, sigma, sigma_in_physical_coordinates=True, FWHM=False, max_kernel_width=32):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.Smooth(2)
        img2 = mytx(img)
        """
        self.sigma = sigma
        self.sigma_in_physical_coordinates = sigma_in_physical_coordinates
        self.FWHM = FWHM
        self.max_kernel_width = max_kernel_width
    
    def __call__(self, *images):
        images = [image.smooth_image(self.sigma,
                                     self.sigma_in_physical_coordinates,
                                     self.FWHM,
                                     self.max_kernel_width) for image in images]
        return images if len(images) > 1 else images[0]


class Crop(BaseTransform):
    def __init__(self, indices=None):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.Crop()
        img2 = mytx(img)
        """
        self.indices = indices
    def __call__(self, *images):
        if self.indices:
            images = [image.crop_indices([i[0] for i in self.indices], 
                                         [i[1] for i in self.indices]) for image in images]
        else:    
            images = [image.crop_image(self.indices) for image in images]
        return images if len(images) > 1 else images[0]


class Resample(BaseTransform):
    
    def __init__(self, resample_params, use_voxels=False, interp_type=1):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('r16'))
        mytx = tx.Resample((2,2))
        img2 = mytx(img)
        """
        self.resample_params = resample_params
        self.use_voxels = use_voxels
        self.interp_type = interp_type
        
    def __call__(self, *images):
        images = [image.resample_image(self.resample_params,
                                       self.use_voxels,
                                       self.interp_type) for image in images]
        return images if len(images) > 1 else images[0]

class Slice(BaseTransform):
    def __init__(self, axis, idx, collapse_strategy=0):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.Slice(0, 120)
        img2 = mytx(img)
        """
        self.axis = axis
        self.idx = idx
        self.collapse_strategy = collapse_strategy
    
    def __call__(self, *images):
        images = [image.slice_image(self.axis, self.idx, self.collapse_strategy) for image in images]
        return images if len(images) > 1 else images[0]


class Pad(BaseTransform):
    def __init__(self, shape=None, pad_width=None, value=0.0):
        """
        import ants
        from nitrain import transforms as tx
        img = ants.image_read(ants.get_data('mni'))
        mytx = tx.Pad((300,300,300))
        img2 = mytx(img)
        """
        self.shape = shape
        self.pad_width = pad_width
        self.value = value
    
    def __call__(self, *images):
        images = [image.pad_image(self.shape, self.pad_width, self.value) for image in images]
        return images if len(images) > 1 else images[0]
