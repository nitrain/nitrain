
import random
import math
import numpy as np
import torch

class RangeNormalize(object):
    """Given min_val: (R, G, B) and max_val: (R,G,B),
    will normalize each channel of the torch.*Tensor to
    the provided min and max values.

    Works by calculating :
        a = (max'-min')/(max-min)
        b = max' - a * max
        new_value = a * value + b
    where min' & max' are given values, 
    and min & max are observed min/max for each channel
    
    Example:
        >>> x = torch.rand(3,5,5)
        >>> rn = RangeNormalize((0,0,10),(1,1,11))
        >>> x_norm = rn(x)

    Also works with just one value for min/max:
        >>> x = torch.rand(3,5,5)
        >>> rn = RangeNormalize(0,1)
        >>> x_norm = rn(x)
    """
    def __init__(self, min_val, max_val, n_channels=1):
        if not isinstance(min_val, list) and not isinstance(min_val, tuple):
            min_val = [min_val]*n_channels
        if not isinstance(max_val, list) and not isinstance(max_val, tuple):
            max_val = [max_val]*n_channels

        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor):
        for t, min_, max_ in zip(tensor, self.min_val, self.max_val):
            _max_val = torch.max(t)
            _min_val = torch.min(t)
            a = (max_-min_)/float(_max_val-_min_val)
            b = max_ - a * _max_val
            t.mul_(a).add_(b)
        return tensor

class StdNormalize(object):
    """Normalize torch tensor to have zero mean and unit std deviation"""

    def __init__(self):
        pass

    def __call__(self, tensor):
        for t in tensor:
            t.sub_(torch.mean(t)).div_(torch.std(t))
        return tensor

class Slice2D(object):

    def __init__(self, axis=0, reject_zeros=True):
        """Take a random 2D slice from a 3D image along 
        a given axis

        Arguments
        ---------
        axis : integer in {0, 1, 2}
            the axis on which to take slices

        reject_zeros : boolean
            whether to reject slices that are all zeros
        """
        self.axis = axis
        self.reject_zeros = reject_zeros

    def __call__(self, x, y=None):
        while True:
            keep_slice  = random.randint(0,x.size(self.axis+1))
            if self.axis == 0:
                slice_x = x[keep_slice,:,:]
                if y is not None:
                    slice_y = y[keep_slice,:,:]
            elif self.axis == 1:
                slice_x = x[:,keep_slice,:]
                if y is not None:
                    slice_y = y[:,keep_slice,:]
            elif self.axis == 2:
                slice_x = x[:,:,keep_slice]
                if y is not None:
                    slice_y = y[:,:,keep_slice]

            if not self.reject_zeros:
                break
            else:
                if y and torch.sum(slice_y) > 0:
                        break
                elif torch.sum(slice_x) > 0:
                        break
        if y:
            return slice_x, slice_y
        else:
            return slice_x

class RandomCrop(object):

    def __init__(self, crop_size):
        """
        Randomly crop a torch tensor

        Arguments
        --------
        size : tuple or list
            dimensions of the crop
        """
        self.crop_size = crop_size

    def __call__(self, x, y=None):
        h_idx = random.randint(0,x.size(1)-self.crop_size[0]+1)
        w_idx = random.randint(0,x.size(2)-self.crop_size[1]+1)
        x = x[:,h_idx:(h_idx+self.crop_size[0]),w_idx:(w_idx+self.crop_size[1])]
        if y is not None:
            y = y[:,h_idx:(h_idx+self.crop_size[0]),w_idx:(w_idx+self.crop_size[1])] 
            return x, y
        else:
            return x

class SpecialCrop(object):

    def __init__(self, crop_size, crop_type=0):
        """
        Perform a special crop - one of the four corners or center crop

        Arguments
        ---------
        crop_type : integer in {0,1,2,3,4}
            0 = center crop
            1 = top left crop
            2 = top right crop
            3 = bottom right crop
            4 = bottom left crop
        """
        if crop_type not in {0, 1, 2, 3, 4}:
            raise ValueError('crop_type must be in {0, 1, 2, 3, 4}')
        self.crop_size = crop_size
        self.crop_type = crop_type
    
    def __call__(self, x, y=None):
        if self.crop_type == 0:
            # center crop
            x_diff  = (x.size(1)-self.crop_size[0])/2.
            y_diff  = (x.size(2)-self.crop_size[1])/2.
            ct_x    = [int(math.ceil(x_diff)),x.size(1)-int(math.floor(x_diff))]
            ct_y    = [int(math.ceil(y_diff)),x.size(2)-int(math.floor(y_diff))]
            indices = [ct_x,ct_y]        
        if self.crop_type == 1:
            # top left crop
            tl_x = [0, self.crop_size[0]]
            tl_y = [0, self.crop_size[1]]
            indices = [tl_x,tl_y]
        elif self.crop_type == 2:
            # top right crop
            tr_x = [0, self.crop_size[0]]
            tr_y = [x.size(2)-self.crop_size[1], x.size(2)]
            indices = [tr_x,tr_y]
        elif self.crop_type == 3:
            # bottom right crop
            br_x = [x.size(1)-self.crop_size[0],x.size(1)]
            br_y = [x.size(2)-self.crop_size[1],x.size(2)]
            indices = [br_x,br_y]
        elif self.crop_type == 4:
            # bottom left crop
            bl_x = [x.size(1)-self.crop_size[0], x.size(1)]
            bl_y = [0, self.crop_size[1]]
            indices = [bl_x,bl_y]
        
        x = x[:,indices[0][0]:indices[0][1],indices[1][0]:indices[1][1]]

        if y is not None:
            y = y[:,indices[0][0]:indices[0][1],indices[1][0]:indices[1][1]]
            return x, y
        else:
            return x


class Pad(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, x, y=None):
        x = x.numpy()
        shape_diffs = [int(np.ceil((i_s - d_s))) for d_s,i_s in zip(x.shape,self.size)]
        shape_diffs = np.maximum(shape_diffs,0)
        pad_sizes = [(int(np.ceil(s/2.)),int(np.floor(s/2.))) for s in shape_diffs]
        x = np.pad(x, pad_sizes, mode='constant')
        if y is not None:
            y = y.numpy()
            y = np.pad(y, pad_sizes, mode='constant')
            return torch.from_numpy(x), torch.from_numpy(y)
        else:
            return torch.from_numpy(x)


class Flip(object):

    def __init__(self, horizontal=True, vertical=False, p=0.5):
        """
        Randomly flip an image horizontally and/or vertically with
        some probability.

        Arguments
        ---------
        horizontal : boolean
            whether to horizontally flip w/ probability p

        vertical : boolean
            whether to vertically flip w/ probability p

        p : float between [0,1]
            probability with which to apply allowed flipping operations
        """
        self.horizontal = horizontal
        self.vertical = vertical
        self.p = p

    def __call__(self, x, y=None):
        x = x.numpy()
        if y is not None:
            y = y.numpy()
        # horizontal flip with p = self.p
        if self.horizontal:
            if random.random() < self.p:
                x = x.swapaxes(2, 0)
                x = x[::-1, ...]
                x = x.swapaxes(0, 2)
                if y is not None:
                    y = y.swapaxes(2, 0)
                    y = y[::-1, ...]
                    y = y.swapaxes(0, 2)
        # vertical flip with p = self.p
        if self.vertical:
            if random.random() < self.p:
                x = x.swapaxes(1, 0)
                x = x[::-1, ...]
                x = x.swapaxes(0, 1)
                if y is not None:
                    y = y.swapaxes(1, 0)
                    y = y[::-1, ...]
                    y = y.swapaxes(0, 1)
        if y is None:
            # must copy because torch doesnt current support neg strides
            return torch.from_numpy(x.copy())
        else:
            return torch.from_numpy(x.copy()),torch.from_numpy(y.copy())


