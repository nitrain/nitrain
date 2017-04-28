
import os
import random
import math
import numpy as np
import torch


class Compose(object):
    """
    Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y=None):
        if y is not None:
            for t in self.transforms:
                x, y = t(x, y)
            return x, y
        else:
            for t in self.transforms:
                x = t(x)
            return x


class ToTensor(object):
    """
    Converts a numpy array to torch.Tensor
    """
    
    def __call__(self, x, y=None):
        x = torch.from_numpy(x)
        if y is not None:
            y = torch.from_numpy(y)
            return x, y
        return x


class ToCuda(object):

    def __init__(self, device=0):
        self.device = device

    def __call__(self, x, y):
        x = x.cuda(self.device)
        if y is not None:
            y = y.cuda(self.device)
            return x, y
        else:
            return x


class ToFile(object):
    """
    Saves an image to file. Useful as the last transform
    when wanting to observe how augmentation/affine transforms
    are affecting the data
    """
    def __init__(self, root, save_format='npy'):
        """Save image to file

        Arguments
        ---------
        root : string
            path to main directory in which images will be saved

        save_format : string in `{'npy', 'jpg', 'png'}
            file format in which to save the sample. Right now, only
            numpy's `npy` format is supported
        """
        self.root = root
        self.save_format = save_format
        self.counter = 0

    def __call__(self, x, y=None):
        np.save(os.path.join(self.root,'x_img-%i.npy'%self.counter), x.numpy())
        if y is not None:
            np.save(os.path.join(self.root,'y_img-%i.npy'%self.counter), y.numpy())
            self.counter += 1
            return x, y
        else:
            self.counter += 1
            return x


class TypeCast(object):

    def __init__(self, dtype='float'):
        self.dtype = dtype

    def __call__(self, x):
        if self.dtype == torch.ByteTensor:
            x = x.byte()
        elif self.dtype == torch.CharTensor:
            x = x.char()
        elif self.dtype == torch.DoubleTensor:
            x = x.double()
        elif self.dtype == torch.FloatTensor:
            x = x.float()
        elif self.dtype == torch.IntTensor:
            x = x.int()
        elif self.dtype == torch.LongTensor:
            x = x.long()
        elif self.dtype == torch.ShortTensor:
            x = x.short()
        else:
            raise Exception('Not a valid Tensor Type')
        return x


class AddChannel(object):
    """
    Adds a dummy channel to an image. 
    This will make an image of size (28, 28) to now be
    of size (1, 28, 28), for example.
    """
    def __call__(self, x, y=None):
        x = x.unsqueeze(2)
        if y is not None:
            y = y.unsqueeze(2)
            return x,y
        return x


class Transpose(object):

    def __init__(self, dim1, dim2):
        self.dim1 = dim1
        self.dim2 = dim2

    def __call__(self, x, y=None):
        x = torch.tranpose(x, self.dim1, self.dim2)
        if y is not None:
            y = torch.tranpose(y, self.dim1, self.dim2)
            return x, y
        else:
            return x


class RangeNormalize(object):
    """
    Given min_val: (R, G, B) and max_val: (R,G,B),
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
    def __init__(self, min_range, max_range, 
        fixed_min=None, fixed_max=None):
        self.min_range = min_range
        self.max_range = max_range
        self.fixed_min = fixed_min
        self.fixed_max = fixed_max

    def __call__(self, x, y=None):
        if self.fixed_min is not None:
            min_val = self.fixed_min
        else:
            min_val = torch.min(x)
        if self.fixed_max is not None:
            max_val = self.fixed_max
        else:
            max_val = torch.max(x)
        if min_val == max_val:
            min_val += 1e-07

        a = (self.max_range - self.min_range) / (max_val - min_val)
        b = self.max_range - a * max_val
        x.mul_(a).add_(b)
        #x.clamp_(self.min_range, self.max_range)
        if y is None:
            return x
        else:
            min_val = torch.min(y)
            max_val = torch.max(y)
            a = (self.max_range - self.min_range) / (max_val - min_val)
            b = self.max_range - a * max_val
            y.mul_(a).add_(b)
            #y.clamp_(self.min_range, self.max_range)
            return x, y


class StdNormalize(object):
    """
    Normalize torch tensor to have zero mean and unit std deviation
    """

    def __init__(self):
        pass

    def __call__(self, x, y=None):
        if y is not None:
            for t, u in zip(x, y):
                t.sub_(torch.mean(t)).div_(torch.std(t))
                u.sub_(torch.mean(u)).div_(torch.std(u))
            return x, y
        else:
            for t in x:
                t.sub_(torch.mean(t)).div_(torch.std(t))
            return x         


class Slice2D(object):

    def __init__(self, axis=0, reject_zeros=False):
        """
        Take a random 2D slice from a 3D image along 
        a given axis. This image should not have a 4th channel dim.

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
            keep_slice  = random.randint(0,x.size(self.axis)-1)
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
                if y is not None and torch.sum(slice_y) > 0:
                        break
                elif torch.sum(slice_x) > 0:
                        break
        if y is not None:
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
        x = x[:, h_idx:(h_idx+self.crop_size[0]),w_idx:(w_idx+self.crop_size[1])]
        if y is not None:
            y = y[:, h_idx:(h_idx+self.crop_size[0]),w_idx:(w_idx+self.crop_size[1])] 
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
        elif self.crop_type == 1:
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
        """
        Pads an image to the given size
        """
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


class RandomFlip(object):

    def __init__(self, h=True, v=False, p=0.5):
        """
        Randomly flip an image horizontally and/or vertically with
        some probability.

        Arguments
        ---------
        h : boolean
            whether to horizontally flip w/ probability p

        v : boolean
            whether to vertically flip w/ probability p

        p : float between [0,1]
            probability with which to apply allowed flipping operations
        """
        self.horizontal = h
        self.vertical = v
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


