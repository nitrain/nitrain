import ntimage as nt
import random

from .base_transform import BaseTransform


class SplitLabels(BaseTransform):
    """
    This transform takes a discrete valued (e.g., segmentation) image
    with no channels (e.g., shape: [128, 128]) and creates separate
    channels for each label. 
    
    For instance, a binary image with 2 classes with shape (128,128) would 
    be turned into an image of shape (128, 128, 2).
    
    Examples
    --------
    >>> import ntimage as nt
    >>> from nitrain import transforms as tx
    >>> img = nt.load(nt.example_data('r16'))
    >>> img = img > img.mean()
    >>> my_tx = tx.SplitLabels()
    >>> img_split = my_tx(img)
    """
    def __init__(self):
        pass
    
    def __call__(self, image, co_image=None):
        # get labels
        labels = image.unique()
        
        # split labels into different images
        image_list = [image == label_value for label_value in labels]
        
        # merge channels
        image_merged = nt.merge(image_list)
        
        return image_merged

class Resample(BaseTransform):
    """
    img = nt.load(nt.example_data('mni'))
    # resample voxel directly (fine if image has even dimensions.. not here though)
    my_tx = ResampleImage((60,60,60))
    img2 = my_tx(img)
    # resample with spacing so you dont have to figure out uneven dimensions
    my_tx2 = ResampleImage((4,4,4), use_spacing=True)
    img3 = my_tx2(img)
    """
    def __init__(self, params, use_spacing=False, interpolation='linear'):
        self.params = params
        self.use_spacing = use_spacing
        self.interpolation = interpolation
        
    def __call__(self, *images):
        new_images = []
        for image in images:
            image = nt.resample(image, self.params, interpolation=self.interpolation, use_spacing=self.use_spacing)
            new_images.append(image)
        
        return new_images if len(new_images) > 1 else new_images[0]

    def __repr__(self):
        return f'tx.Resample({self.params}, {self.use_spacing}, "{self.interpolation}")'

class ResampleToTarget(BaseTransform):
    """
    img = nt.load(nt.example_data('mni'))
    img2 = img.clone().resample_image((4,4,4))
    my_tx = ResampleImageToTarget(img2)
    img3 = my_tx(img)
    """
    def __init__(self, target, interpolation='linear'):
        self.target = target
        self.interpolation = 0 if interpolation != 'nearest_neighbor' else 1 
    
    def __call__(self, image):
        image = nt.resample(image, self.target, self.interpolation)
        return image


class Reorient(BaseTransform):
    """
    Reorient an image.
    
    Images are oriented along three axes:
    - Right (R) to Left (L)
    - Inferior (I) to Superior (S) 
    - Anterior (A) to Posterior (P)
    
    An image orientation consists of three letters - one from each
    of the three axes - with the letter for each axes determining
    where the indexing starts. Orientation is important for slicing.
    """
    def __init__(self, orientation='RAS'):
        self.orientation = orientation
    
    def __call__(self, image, co_image=None):
        image = nt.reorient(image, self.orientation)
        
        if co_image is not None:
            co_image = nt.reorient(co_image, self.orientation)
            return image, co_image

        return image

class Slice(BaseTransform):
    """
    Slice a 3D image into 2D. 
    """
    def __init__(self, axis, idx):
        self.axis = axis
        self.idx = idx
    
    def __call__(self, image):
        if self.idx is None:
            new_image = [image.slice_image(self.axis, idx, 1) for idx in range(image.shape[self.axis])]
        else:
            new_image = image.slice_image(self.axis, self.idx, 1)
        return new_image


class RandomSlice(BaseTransform):
    """
    Randomly slice a 3D image into 2D. 
    
    """
    def __init__(self, axis, allow_blank=True):
        self.axis = axis
        self.allow_blank = allow_blank
    
    def __call__(self, image):
        if not self.allow_blank:
            image = image.crop_image()
        
        idx = random.sample(range(image.shape[self.axis]), 1)[0]
        new_image = image.slice_image(self.axis, idx)
        
        return new_image


class Crop(BaseTransform):
    """
    Crop an image to remove all blank space around the brain or
    crop based on specified indices.
    """
    def __init__(self, lower_indices=None, upper_indices=None):
        self.lower_indices = lower_indices
        self.upper_indices = upper_indices
    
    def __call__(self, image):
        if self.lower_indices and self.upper_indices:
            new_image = image.crop_indices(self.lower_indices,
                                           self.upper_indices)
        else:
            new_image = image.crop_image()
        return new_image


class RandomCrop(BaseTransform):
    """
    Randomly crop an image of the specified size
    
    Examples
    --------
    >>> import ntimage as nt
    >>> from nitrain import transforms as tx
    >>> mni = nt.load(nt.example_data('mni'))
    >>> my_tx = tx.RandomCrop(size=(30,30,30))
    >>> mni_crop = my_tx(mni)
    >>> mni_crop.plot(domain_image_map=mni)
    >>> mni_orig = mni_crop.decrop_image(mni) # put image back
    """
    def __init__(self, size):
        self.size = size
    
    def __call__(self, image):
        size = self.size
        if isinstance(size, int):
            size = tuple([size for _ in range(image.dimension)])
            
        lower_indices = [random.sample(range(0, image.shape[i]-size[0]), 1)[0] for i in range(len(size))]
        upper_indices = [lower_indices[i] + size[i] for i in range(len(size))]
            
        new_image = image.crop_indices(lower_indices,
                                       upper_indices)
        return new_image


class Pad(BaseTransform):
    """
    Pad an image to a specified shape or by a specified amount.
    
    Example
    -------
    >>> import ntimage as nt
    >>> from nitrain import transforms as tx
    >>> mni = nt.load(nt.example_data('mni'))
    >>> my_tx = tx.Pad((220,220,220))
    >>> mni_pad = my_tx(mni)
    """
    
    def __init__(self, shape=None, width=None, value=0.0):
        if shape is None and width is None:
            raise Exception('Either shape or width must be supplied to Pad transform.')
        self.shape = shape
        self.width = width
        self.value = value
    
    def __call__(self, image):
        new_image = image.pad_image(shape=self.shape,
                                    pad_width=self.width,
                                    value=self.value)
        return new_image