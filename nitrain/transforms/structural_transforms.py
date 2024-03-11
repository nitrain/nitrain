from .base_transform import BaseTransform

class ExpandDims(BaseTransform):
    pass

class SwapAxes:
    pass


class Slice(BaseTransform):
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

class Crop(BaseTransform):
    pass

class Pad(BaseTransform):
    pass
