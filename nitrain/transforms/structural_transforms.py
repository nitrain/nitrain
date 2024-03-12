import ants

from .base_transform import BaseTransform


class Resample(BaseTransform):
    """
    img = ants.image_read(ants.get_ants_data('mni'))
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
        self.interpolation = 0 if interpolation != 'nearest_neighbor' else 1 
    
    def __call__(self, image):
        image = ants.resample_image(image, self.params, not self.use_spacing, self.interpolation)
        return image


class ResampleToTarget(BaseTransform):
    """
    img = ants.image_read(ants.get_ants_data('mni'))
    img2 = img.clone().resample_image((4,4,4))
    my_tx = ResampleImageToTarget(img2)
    img3 = my_tx(img)
    """
    def __init__(self, target, interpolation='linear'):
        self.target = target
        self.interpolation = 0 if interpolation != 'nearest_neighbor' else 1 
    
    def __call__(self, image):
        image = ants.resample_image_to_target(image, self.target, self.interpolation)
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
    
    def __call__(self, image):
        image = ants.reorient_image2(image, self.orientation)
        

class Slice(BaseTransform):
    pass

class Crop(BaseTransform):
    pass

class Pad(BaseTransform):
    pass

class ExtractPatches:
    """
            image_patches = list()
            for ii in range(len(images)):
                image_patches.append(antspynet.extract_image_patches(images[ii], 
                                                         patch_size=patch_size,
                                                         max_number_of_patches=number_of_patches_per_image,
                                                         mask_image=seg_mask,
                                                         random_seed=random_seed,
                                                         return_as_array=True))
            seg_patches = antspynet.extract_image_patches(seg, 
                                                          patch_size=patch_size,
                                                          max_number_of_patches=number_of_patches_per_image,
                                                          mask_image=seg_mask,
                                                          random_seed=random_seed,
                                                          return_as_array=True)
    """
    pass
