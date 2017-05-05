"""
Transforms very specific to images such as 
color, lighting, contrast, brightness, etc transforms
"""
import torch as th
import random

from ..utils import th_zeros_like

class Lighting(object):
    """
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


def _blend(img1, img2, alpha):
    """
    local function blend(img1, img2, alpha)
       return img1:mul(alpha):add(1 - alpha, img2)
    end
    """
    return img1.mul(alpha).add(1 - alpha, img2)


class Grayscale(object):
    """
    local function grayscale(dst, img)
       dst:resizeAs(img)
       dst[1]:zero()
       dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
       dst[2]:copy(dst[1])
       dst[3]:copy(dst[1])
       return dst
    end

    Example:
    >>> import PIL.Image
    >>> import numpy as np
    >>> import torch as th
    >>> import matplotlib.pyplot as plt
    >>> %matplotlib inline
    >>> img = PIL.Image.open('/users/ncullen/desktop/projects/torchsample/tests/image.jpg')
    >>> img = th.from_numpy(np.rollaxis(np.asarray(img),2)).float()
    >>> gs = Grayscale(keep_channels=True)(img)
    >>> PIL.Image.fromarray(np.uint8(gs.permute(1,2,0).numpy()))
    """
    def __init__(self, keep_channels=False):
        self.keep_channels = keep_channels
        if keep_channels:
            self.channels = 3
        else:
            self.channels = 1

    def __call__(self, x, y=None):
        x_dst = x.new(th.Size([self.channels])+x.size()[1:]).zero_()
        x_dst[0].add_(0.299,x[0]).add_(0.587,x[1]).add_(0.114,x[2])
        if self.keep_channels:
            x_dst[1].copy_(x_dst[0])
            x_dst[2].copy_(x_dst[1])

        if y is not None:
            y_dst = y.new(th.Size([self.channels])+x.size()[1:]).zero_()
            y_dst[0].add_(0.299,y[0]).add_(0.587,y[1]).add_(0.114,y[2])
            if self.keep_channels:
                y_dst[1].copy_(y_dst[0])
                y_dst[2].copy_(y_dst[1])

            return x_dst, y_dst
        return x_dst


class Saturation(object):
    """
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
        alpha = 1.0 + random.uniform(-self.var, self.var)
        x = _blend(x, x_gs, alpha)
        if y is not None:
            y_gs = th_zeros_like(y)
            # use the same alpha
            y = _blend(y, y_gs, alpha)
            return x, y
        return x


class Constrast(object):
    """
    function M.Contrast(var)
       local gs

       return function(input)
          gs = gs or input.new()
          grayscale(gs, input)
          gs:fill(gs[1]:mean())

          local alpha = 1.0 + torch.uniform(-var, var)
          blend(input, gs, alpha)
          return input
       end
    end
    """
    def __init__(self, var):
        self.var = var

    def __call__(self, x, y=None):
        x_gs = Grayscale(keep_channels=True)(x)
        x_gs.fill_(x_gs[1].mean())

        alpha = 1.0 + random.uniform(-self.var, self.var)
        x = _blend(x, x_gs, alpha)

        if y is not None:
            y_gs = Grayscale(keep_channels=True)(y)
            y_gs.fill_(y_gs[1].mean())
            y = _blend(y, y_gs, alpha)
            return x, y
        return x






