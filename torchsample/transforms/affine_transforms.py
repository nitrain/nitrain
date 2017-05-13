"""
Affine transforms implemented on torch tensors, and
requiring only one interpolation
"""

import math
import random
import torch as th

from ..utils import th_affine2d, th_random_choice


class RandomAffine(object):

    def __init__(self, 
                 rotation_range=None, 
                 translation_range=None,
                 shear_range=None, 
                 zoom_range=None,
                 interp='bilinear',
                 lazy=False):
        """
        Perform an affine transforms with various sub-transforms, using
        only one interpolation and without having to instantiate each
        sub-transform individually.

        Arguments
        ---------
        rotation_range : one integer or float
            image will be rotated between (-degrees, degrees) degrees

        translation_range : a float or a tuple/list w/ 2 floats between [0, 1)
            first value:
                image will be horizontally shifted between 
                (-height_range * height_dimension, height_range * height_dimension)
            second value:
                Image will be vertically shifted between 
                (-width_range * width_dimension, width_range * width_dimension)

        shear_range : float
            radian bounds on the shear transform

        zoom_range : list/tuple with two floats between [0, infinity).
            first float should be less than the second
            lower and upper bounds on percent zoom. 
            Anything less than 1.0 will zoom in on the image, 
            anything greater than 1.0 will zoom out on the image.
            e.g. (0.7, 1.0) will only zoom in, 
                 (1.0, 1.4) will only zoom out,
                 (0.7, 1.4) will randomly zoom in or out

        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform
            ProTip : use 'nearest' for discrete images (e.g. segmentations)
                    and use 'constant' for continuous images

        fill_value : float
            the value to fill the empty space with if fill_mode='constant'

        target_fill_mode : same as fill_mode, but for target image

        target_fill_value : same as fill_value, but for target image

        """
        self.transforms = []
        if rotation_range is not None:
            rotation_tform = RandomRotate(rotation_range, lazy=True)
            self.transforms.append(rotation_tform)

        if translation_range is not None:
            translation_tform = RandomTranslate(translation_range, lazy=True)
            self.transforms.append(translation_tform)

        if shear_range is not None:
            shear_tform = RandomShear(shear_range, lazy=True)
            self.transforms.append(shear_tform) 

        if zoom_range is not None:
            zoom_tform = RandomZoom(zoom_range, lazy=True)
            self.transforms.append(zoom_tform)

        self.interp = interp
        self.lazy = lazy

        if len(self.transforms) == 0:
            raise Exception('Must give at least one transform parameter')

    def __call__(self, *inputs):
        # collect all of the lazily returned tform matrices
        tform_matrix = self.transforms[0](inputs[0])
        for tform in self.transforms[1:]:
            tform_matrix = tform_matrix.mm(tform(inputs[0])) 
        self.tform_matrix = tform_matrix

        if self.lazy:
            return tform_matrix
        else:
            outputs = Affine(tform_matrix,
                             interp=self.interp)(*inputs)
            return outputs


class Affine(object):

    def __init__(self, 
                 tform_matrix,
                 interp='bilinear'):
        """
        Perform an affine transforms with various sub-transforms, using
        only one interpolation and without having to instantiate each
        sub-transform individually.

        Arguments
        ---------
        tform_matrix : a 2x3 or 3x3 matrix

        """
        self.tform_matrix = tform_matrix
        self.interp = interp

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        outputs = []
        for idx, _input in enumerate(inputs):
            input_tf = th_affine2d(_input,
                                   self.tform_matrix,
                                   mode=interp[idx])
            outputs.append(input_tf)
        return outputs if idx > 1 else outputs[0]


class AffineCompose(object):

    def __init__(self, 
                 transforms,
                 interp='bilinear'):
        """
        Apply a collection of explicit affine transforms to an input image,
        and to a target image if necessary

        Arguments
        ---------
        transforms : list or tuple
            each element in the list/tuple should be an affine transform.
            currently supported transforms:
                - Rotate()
                - Translate()
                - Shear()
                - Zoom()

        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform

        fill_value : float
            the value to fill the empty space with if fill_mode='constant'

        """
        self.transforms = transforms
        self.interp = interp
        # set transforms to lazy so they only return the tform matrix
        for t in self.transforms:
            t.lazy = True

    def __call__(self, *inputs):
        # collect all of the lazily returned tform matrices
        tform_matrix = self.transforms[0](inputs[0])
        for tform in self.transforms[1:]:
            tform_matrix = tform_matrix.mm(tform(inputs[0])) 

        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        outputs = []
        for idx, _input in enumerate(inputs):
            input_tf = th_affine2d(_input,
                                   tform_matrix,
                                   mode=interp[idx])
            outputs.append(input_tf)
        return outputs if idx > 1 else outputs[0]


class RandomRotate(object):

    def __init__(self, 
                 rotation_range,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly rotate an image between (-degrees, degrees). If the image
        has multiple channels, the same rotation will be applied to each channel.

        Arguments
        ---------
        rotation_range : integer or float
            image will be rotated between (-degrees, degrees) degrees

        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform

        fill_value : float
            the value to fill the empty space with if fill_mode='constant'

        lazy    : boolean
            if false, perform the transform on the tensor and return the tensor
            if true, only create the affine transform matrix and return that
        """
        self.rotation_range = rotation_range
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        degree = random.uniform(-self.rotation_range, self.rotation_range)

        if self.lazy:
            return Rotate(degree, lazy=True)(inputs[0])
        else:
            outputs = Rotate(degree,
                             interp=self.interp)(*inputs)
            return outputs


class RandomChoiceRotate(object):

    def __init__(self, 
                 values,
                 probs=None,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly rotate an image between (-degrees, degrees). If the image
        has multiple channels, the same rotation will be applied to each channel.

        Arguments
        ---------
        rotation_range : integer or float
            image will be rotated between (-degrees, degrees) degrees

        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform

        fill_value : float
            the value to fill the empty space with if fill_mode='constant'

        lazy    : boolean
            if false, perform the transform on the tensor and return the tensor
            if true, only create the affine transform matrix and return that
        """
        if isinstance(values, (list, tuple)):
            values = th.FloatTensor(values)
        self.values = values
        if probs is None:
            probs = th.ones(len(values)) / len(values)
        else:
            if abs(1.0-sum(probs)) > 1e-3:
                raise ValueError('Probs must sum to 1')
        self.probs = probs
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        degree = th_random_choice(self.values, p=self.probs)

        if self.lazy:
            return Rotate(degree, lazy=True)(inputs[0])
        else:
            outputs = Rotate(degree,
                             interp=self.interp)(*inputs)
            return outputs


class Rotate(object):

    def __init__(self, 
                 value,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly rotate an image between (-degrees, degrees). If the image
        has multiple channels, the same rotation will be applied to each channel.

        Arguments
        ---------
        rotation_range : integer or float
            image will be rotated between (-degrees, degrees) degrees

        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform

        fill_value : float
            the value to fill the empty space with if fill_mode='constant'

        lazy    : boolean
            if false, perform the transform on the tensor and return the tensor
            if true, only create the affine transform matrix and return that
        """
        self.value = value
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        theta = math.pi / 180 * self.value
        rotation_matrix = th.FloatTensor([[math.cos(theta), -math.sin(theta), 0],
                                          [math.sin(theta), math.cos(theta), 0],
                                          [0, 0, 1]])
        if self.lazy:
            return rotation_matrix
        else:
            outputs = []
            for idx, _input in enumerate(inputs):
                input_tf = th_affine2d(_input,
                                       rotation_matrix,
                                       mode=interp[idx],
                                       center=True)
                outputs.append(input_tf)
            return outputs if idx > 1 else outputs[0]


class RandomTranslate(object):

    def __init__(self, 
                 translation_range,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly translate an image some fraction of total height and/or
        some fraction of total width. If the image has multiple channels,
        the same translation will be applied to each channel.

        Arguments
        ---------
        translation_range : two floats between [0, 1) 
            first value:
                fractional bounds of total height to shift image
                image will be horizontally shifted between 
                (-height_range * height_dimension, height_range * height_dimension)
            second value:
                fractional bounds of total width to shift image 
                Image will be vertically shifted between 
                (-width_range * width_dimension, width_range * width_dimension)

        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform

        fill_value : float
            the value to fill the empty space with if fill_mode='constant'

        lazy    : boolean
            if false, perform the transform on the tensor and return the tensor
            if true, only create the affine transform matrix and return that
        """
        if isinstance(translation_range, float):
            translation_range = (translation_range, translation_range)
        self.height_range = translation_range[0]
        self.width_range = translation_range[1]
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        # height shift
        random_height = random.uniform(-self.height_range, self.height_range)
        # width shift
        random_width = random.uniform(-self.width_range, self.width_range)

        if self.lazy:
            return Translate([random_height, random_width], 
                             lazy=True)(inputs[0])
        else:
            outputs = Translate([random_height, random_width],
                                 interp=self.interp)(*inputs)
            return outputs


class RandomChoiceTranslate(object):

    def __init__(self,
                 values,
                 probs=None,
                 interp='bilinear',
                 lazy=False):
        if isinstance(values, (list, tuple)):
            values = th.FloatTensor(values)
        self.values = values
        if probs is None:
            probs = th.ones(len(values)) / len(values)
        else:
            if abs(1.0-sum(probs)) > 1e-3:
                raise ValueError('Probs must sum to 1')
        self.probs = probs
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        random_height = th_random_choice(self.values, p=self.probs)
        random_width = th_random_choice(self.values, p=self.probs)

        if self.lazy:
            return Translate([random_height, random_width],
                             lazy=True)(inputs[0])
        else:
            outputs = Translate([random_height, random_width],
                                interp=self.interp)(*inputs)
            return outputs


class Translate(object):

    def __init__(self, 
                 value, 
                 interp='bilinear',
                 lazy=False):
        """
        Arguments
        ---------
        value : float or 2-tuple of float
            if single value, both horizontal and vertical translation
            will be this value * total height/width. Thus, value should
            be a fraction of total height/width with range (-1, 1)
        """
        if not isinstance(value, (tuple,list)):
            value = (value, value)

        if value[0] > 1 or value[0] < -1:
            raise ValueError('Translation must be between -1 and 1')
        if value[1] > 1 or value[1] < -1:
            raise ValueError('Translation must be between -1 and 1')

        self.height_range = value[0]
        self.width_range = value[1]
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        tx = self.height_range * inputs[0].size(1)
        ty = self.width_range * inputs[0].size(2)

        translation_matrix = th.FloatTensor([[1, 0, tx],
                                             [0, 1, ty],
                                             [0, 0, 1]])
        if self.lazy:
            return translation_matrix
        else:
            outputs = []
            for idx, _input in enumerate(inputs):
                input_tf = th_affine2d(_input,
                                       translation_matrix,
                                       mode=interp[idx],
                                       center=True)
                outputs.append(input_tf)
            return outputs if idx > 1 else outputs[0]


class RandomShear(object):

    def __init__(self, 
                 shear_range,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly shear an image with radians (-shear_range, shear_range)

        Arguments
        ---------
        shear_range : float
            radian bounds on the shear transform
        
        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform

        fill_value : float
            the value to fill the empty space with if fill_mode='constant'

        lazy    : boolean
            if false, perform the transform on the tensor and return the tensor
            if true, only create the affine transform matrix and return that
        """
        self.shear_range = shear_range
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        shear = random.uniform(-self.shear_range, self.shear_range)
        if self.lazy:
            return Shear(shear, 
                         lazy=True)(inputs[0])
        else:
            outputs = Shear(shear,
                            interp=self.interp)(*inputs)
            return outputs


class RandomChoiceShear(object):

    def __init__(self,
                 values,
                 probs=None,
                 interp='bilinear',
                 lazy=False):
        if isinstance(values, (list, tuple)):
            values = th.FloatTensor(values)
        self.values = values
        if probs is None:
            probs = th.ones(len(values)) / len(values)
        else:
            if abs(1.0-sum(probs)) > 1e-3:
                raise ValueError('Probs must sum to 1')
        self.probs = probs
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        shear = th_random_choice(self.values, p=self.probs)

        if self.lazy:
            return Shear(shear, 
                         lazy=True)(inputs[0])
        else:
            outputs = Shear(shear,
                            interp=self.interp)(*inputs)
            return outputs 


class Shear(object):

    def __init__(self,
                 value,
                 interp='bilinear',
                 lazy=False):
        self.value = value
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        theta = (math.pi * self.value) / 180
        shear_matrix = th.FloatTensor([[1, -math.sin(theta), 0],
                                        [0, math.cos(theta), 0],
                                        [0, 0, 1]])
        if self.lazy:
            return shear_matrix
        else:
            outputs = []
            for idx, _input in enumerate(inputs):
                input_tf = th_affine2d(_input,
                                       shear_matrix,
                                       mode=interp[idx],
                                       center=True)
                outputs.append(input_tf)
            return outputs if idx > 1 else outputs[0]


class RandomZoom(object):

    def __init__(self, 
                 zoom_range,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly zoom in and/or out on an image 

        Arguments
        ---------
        zoom_range : tuple or list with 2 values, both between (0, infinity)
            lower and upper bounds on percent zoom. 
            Anything less than 1.0 will zoom in on the image, 
            anything greater than 1.0 will zoom out on the image.
            e.g. (0.7, 1.0) will only zoom in, 
                 (1.0, 1.4) will only zoom out,
                 (0.7, 1.4) will randomly zoom in or out

        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform

        fill_value : float
            the value to fill the empty space with if fill_mode='constant'

        lazy    : boolean
            if false, perform the transform on the tensor and return the tensor
            if true, only create the affine transform matrix and return that
        """
        if not isinstance(zoom_range, list) and not isinstance(zoom_range, tuple):
            raise ValueError('zoom_range must be tuple or list with 2 values')
        self.zoom_range = zoom_range
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        zx = random.uniform(self.zoom_range[0], self.zoom_range[1])
        zy = random.uniform(self.zoom_range[0], self.zoom_range[1])

        if self.lazy:
            return Zoom([zx, zy], lazy=True)(inputs[0])
        else:
            outputs = Zoom([zx, zy], 
                           interp=self.interp)(*inputs)
            return outputs


class RandomChoiceZoom(object):

    def __init__(self, 
                 values,
                 probs=None,
                 interp='bilinear',
                 lazy=False):
        """
        Randomly zoom in and/or out on an image 

        Arguments
        ---------
        zoom_range : tuple or list with 2 values, both between (0, infinity)
            lower and upper bounds on percent zoom. 
            Anything less than 1.0 will zoom in on the image, 
            anything greater than 1.0 will zoom out on the image.
            e.g. (0.7, 1.0) will only zoom in, 
                 (1.0, 1.4) will only zoom out,
                 (0.7, 1.4) will randomly zoom in or out

        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform

        fill_value : float
            the value to fill the empty space with if fill_mode='constant'

        lazy    : boolean
            if false, perform the transform on the tensor and return the tensor
            if true, only create the affine transform matrix and return that
        """
        if isinstance(values, (list, tuple)):
            values = th.FloatTensor(values)
        self.values = values
        if probs is None:
            probs = th.ones(len(values)) / len(values)
        else:
            if abs(1.0-sum(probs)) > 1e-3:
                raise ValueError('Probs must sum to 1')
        self.probs = probs
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        zx = th_random_choice(self.values, p=self.probs)
        zy = th_random_choice(self.values, p=self.probs)

        if self.lazy:
            return Zoom([zx, zy], lazy=True)(inputs[0])
        else:
            outputs = Zoom([zx, zy], 
                           interp=self.interp)(*inputs)
            return outputs


class Zoom(object):

    def __init__(self,
                 value,
                 interp='bilinear',
                 lazy=False):
        """
        Arguments
        ---------
        value : float
            Fractional zoom.
            =1 : no zoom
            >1 : zoom-in (value-1)%
            <1 : zoom-out (1-value)%

        lazy: boolean
            If true, just return transformed
        """

        if not isinstance(value, (tuple,list)):
            value = (value, value)
        self.value = value
        self.interp = interp
        self.lazy = lazy

    def __call__(self, *inputs):
        if not isinstance(self.interp, (tuple,list)):
            interp = [self.interp]*len(inputs)
        else:
            interp = self.interp

        zx, zy = self.value
        zoom_matrix = th.FloatTensor([[zx, 0, 0],
                                      [0, zy, 0],
                                      [0, 0,  1]])        

        if self.lazy:
            return zoom_matrix
        else:
            outputs = []
            for idx, _input in enumerate(inputs):
                input_tf = th_affine2d(_input,
                                       zoom_matrix,
                                       mode=interp[idx],
                                       center=True)
                outputs.append(input_tf)
            return outputs if idx > 1 else outputs[0]


