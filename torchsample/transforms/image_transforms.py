"""
Transforms very specific to images such as 
color, lighting, contrast, brightness, etc transforms
"""
import torch as th
import random

from ..utils import th_zeros_like

def _blend(img1, img2, alpha):
    """
    Takes weighted sum of two images

    Example:
    >>> # img will be all 0.4's
    >>> img = _blend(th.ones(5,5),th.zeros(5,5),0.4) 
    """
    return img1.mul(alpha).add(1 - alpha, img2)


class Grayscale(object):
    """
    Convert RGB image to grayscale

    Example:
    >>> import PIL.Image
    >>> import numpy as np
    >>> import torch as th
    >>> img = PIL.Image.open('/users/ncullen/desktop/projects/torchsample/tests/image.jpg')
    >>> img = th.from_numpy(np.rollaxis(np.asarray(img),2)).float()/255.
    >>> gs = Grayscale(keep_channels=True)(img)
    >>> PIL.Image.fromarray(np.uint8(gs.permute(1,2,0).numpy()*255.))
    """
    def __init__(self, keep_channels=False):
        """
        Convert RGB image to grayscale

        Arguments
        ---------
        keep_channels : boolean
            If true, will keep all 3 channels and they will be the same
            If false, will just return 1 grayscale channel
        """
        self.keep_channels = keep_channels
        if keep_channels:
            self.channels = 3
        else:
            self.channels = 1

    def __call__(self, x, y=None):
        x_dst = x[0]*0.299 + x[1]*0.587 + x[2]*0.114
        x_gs = x_dst.repeat(self.channels,1,1)

        if y is not None:
            y_dst = y[0]*0.299 + y[1]*0.587 + y[2]*0.114
            y_gs= y_dst.repeat(self.channels,1,1)
            return x_gs, y_gs
        return x_gs


class Lighting(object):
    """
    NOT TESTED NOT TESTED NOT TESTED

    function M.Lighting(alphastd, eigval, eigvec)
       return function(input)
          if alphastd == 0 then
             return input
          end

          local alpha = torch.Tensor(3):normal(0, alphastd)
          local rgb = eigvec:clone()
             :cmul(alpha:view(1, 3):expand(3, 3))
             :cmul(eigval:view(1, 3):expand(3, 3))
             :sum(2)
             :squeeze()

          input = input:clone()
          for i=1,3 do
             input[i]:add(rgb[i])
          end
          return input
       end
    end
    """

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, x, y=None):
        if self.alphastd == 0:
            if y is not None:
                return x, y
            return x

        alpha = th.Tensor(3).normal_(0, self.alphastd)
        rgb = self.eigvec.clone().cmul(alpha.view(1,3).expand(3,3))
        rgb = rgb.cmul(self.eigval.view(1,3).expand(3,3))
        rgb = rgb.sum(2).squeeze()

        x = x.clone()
        for i in range(3):
            x[i].add_(rgb[i])
        if y is not None:
            y = y.clone()
            for i in range(3):
                y[i].add_(rgb[i])
            return x, y
        return x


class Saturation(object):
    """
    NOT TESTED NOT TESTED NOT TESTED

    function M.Saturation(var)
       local gs

       return function(input)
          gs = gs or input.new()
          grayscale(gs, input)

          local alpha = 1.0 + torch.uniform(-var, var)
          blend(input, gs, alpha)
          return input
       end
    end

    >>> import PIL.Image
    >>> import numpy as np
    >>> import torch as th
    >>> import matplotlib.pyplot as plt
    >>> %matplotlib inline
    >>> img = PIL.Image.open('/users/ncullen/desktop/projects/torchsample/tests/image.jpg')
    >>> img = th.from_numpy(np.rollaxis(np.asarray(img),2)).float()
    >>> gs = Saturation(0.5)(img)
    >>> PIL.Image.fromarray(np.uint8(gs.permute(1,2,0).numpy()))
    """
    def __init__(self, var):
        self.var = var

    def __call__(self, x, y=None):
        x_gs = Grayscale(keep_channels=True)(x)
        alpha = 1.0 + random.uniform(-self.var, self.var)
        x = _blend(x, x_gs, alpha)
        if y is not None:
            y_gs = Grayscale(keep_channels=True)(y)
            y = _blend(y, y_gs, alpha)
            return x, y
        return x


class Brightness(object):
    """
    NOT TESTED NOT TESTED NOT TESTED

    function M.Brightness(var)
       local gs

       return function(input)
          gs = gs or input.new()
          gs:resizeAs(input):zero()

          local alpha = 1.0 + torch.uniform(-var, var)
          blend(input, gs, alpha)
          return input
       end
    end

    >>> import PIL.Image
    >>> import numpy as np
    >>> import torch as th
    >>> import matplotlib.pyplot as plt
    >>> %matplotlib inline
    >>> img = PIL.Image.open('/users/ncullen/desktop/projects/torchsample/tests/image.jpg')
    >>> img = th.from_numpy(np.rollaxis(np.asarray(img),2)).float()
    >>> gs = Brightness(0.5)(img)
    >>> PIL.Image.fromarray(np.uint8(gs.permute(1,2,0).numpy()))

    """
    def __init__(self, var):
        self.var = var

    def __call__(self, x, y=None):
        x_gs = th_zeros_like(x)
        alpha = 1.0 + self.var#random.uniform(-self.var, self.var)
        x = _blend(x, x_gs, alpha)
        if y is not None:
            y_gs = th_zeros_like(y)
            # use the same alpha
            y = _blend(y, y_gs, alpha)
            return x, y
        return x


class Contrast(object):
    """
    NOT TESTED NOT TESTED NOT TESTED

    Contrast is adjusted independently for each channel of each image.

    For each channel, this Op computes the mean of the image pixels 
    in the channel and then adjusts each component x of each pixel to 
    (x - mean) * contrast_factor + mean.

    >>> import PIL.Image
    >>> import numpy as np
    >>> import torch as th
    >>> import matplotlib.pyplot as plt
    >>> %matplotlib inline
    >>> img = PIL.Image.open('/users/ncullen/desktop/projects/torchsample/tests/image.jpg')
    >>> x = th.from_numpy(np.rollaxis(np.asarray(img),2)).float()

    """
    def __init__(self, var):
        self.var = var

    def __call__(self, x, y=None):
        channel_means = x.mean(1)
        for i in range(2,x.dim()):
            channel_means = channel_means.mean(i)
        channel_means = channel_means.expand_as(x)

        x = (x - channel_means) * self.var + channel_means
        if y is not None:
            channel_means = x.mean(1).mean(2).expand_as(x)
            x = (x - channel_means) * self.var + channel_means          


class Gamma(object):
    """
    !! WORKS WORKS WORKS !!

    Performs Gamma Correction on the input image. Also known as 
    Power Law Transform. This function transforms the input image 
    pixelwise according 
    to the equation Out = In**gamma after scaling each 
    pixel to the range 0 to 1.
    """
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, x, y=None):
        x = th.pow(x, self.gamma)
        if y is not None:
            y = th.pow(y, self.gamma)
            return x, y
        return x




