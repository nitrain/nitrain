"""
Transforms very specific to images such as 
color, lighting, contrast, brightness, etc transforms

NOTE: Most of these transforms assume your image intensity
is between 0 and 1, and are torch tensors (NOT numpy or PIL)
"""
import torch as th
import random


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


class Gamma(object):
    """
    Performs Gamma Correction on the input image. Also known as 
    Power Law Transform. This function transforms the input image 
    pixelwise according 
    to the equation Out = In**gamma after scaling each 
    pixel to the range 0 to 1.

    """
    def __init__(self, value):
        """
        Perform Gamma correction

        Arguments
        ---------
        gamma : float
            <1 : image will tend to be lighter
            =1 : image will stay the same
            >1 : image will tend to be darker
        """
        self.value = value

    def __call__(self, x, y=None):
        x = th.pow(x, self.value)
        if y is not None:
            y = th.pow(y, self.value)
            return x, y
        return x


class Brightness(object):
    """
    Alter the Brightness of an image
    """
    def __init__(self, value):
        """
        Arguments
        ---------
        value : brightness factor
            =-1 = completely black
            <0 = darker
            0 = no change
            >0 = brighter
            =1 = completely white
        """
        self.value = max(min(value,1.0),-1.0)

    def __call__(self, x, y=None):
        x = th.clamp(x.float().add(self.value).type(x.type()), 0, 1)
        if y is not None:
            y = th.clamp(y.float().add(self.value).type(y.type()), 0, 1)
            return x, y
        return x


class Saturation(object):
    """
    Alter the Saturation of image
    """
    def __init__(self, value):
        """
        Arguments
        ---------
        value : float
            =-1 : gray
            <0 : colors are more muted
            =0 : image stays the same
            >0 : colors are more pure
            =1 : most saturated
        """
        self.value = max(min(value,1.0),-1.0)

    def __call__(self, x, y=None):
        x_gs = Grayscale(keep_channels=True)(x)
        alpha = 1.0 + self.value
        x = th.clamp(_blend(x, x_gs, alpha),0,1)
        if y is not None:
            y_gs = Grayscale(keep_channels=True)(y)
            y = th.clamp(_blend(y, y_gs, alpha),0,1)
            return x, y
        return x


class Contrast(object):
    """
    Contrast is adjusted independently for each channel of each image.

    For each channel, this Op computes the mean of the image pixels 
    in the channel and then adjusts each component x of each pixel to 
    (x - mean) * contrast_factor + mean.
    """
    def __init__(self, value):
        """
        Arguments
        ---------
        value : float
            smaller value: less contrast
            ZERO: channel means
            larger positive value: greater contrast
            larger negative value: greater inverse contrast
            
    
        """
        self.value = value

    def __call__(self, x, y=None):
        channel_means = x.mean(1).mean(2)
        channel_means = channel_means.expand_as(x)
        x = th.clamp((x - channel_means) * self.value + channel_means,0,1)

        if y is not None:
            channel_means = y.mean(1).mean(2).expand_as(y)
            y = th.clamp((y - channel_means) * self.value + channel_means,0,1)       
            return x, y
        return x


def rgb_to_hsv(x):
    """
    Convert from RGB to HSV
    """
    hsv = th.zeros(*x.size())
    c_min = x.min(0)
    c_max = x.max(0)

    delta = c_max[0] - c_min[0]

    # set H
    r_idx = c_max[1].eq(0)
    hsv[0][r_idx] = ((x[1][r_idx] - x[2][r_idx]) / delta[r_idx]) % 6
    g_idx = c_max[1].eq(1)
    hsv[0][g_idx] = 2 + ((x[2][g_idx] - x[0][g_idx]) / delta[g_idx])
    b_idx = c_max[1].eq(2)
    hsv[0][b_idx] = 4 + ((x[0][b_idx] - x[1][b_idx]) / delta[b_idx])
    hsv[0] = hsv[0].mul(60)

    # set S
    hsv[1] = delta / c_max[0]

    # set V - good
    hsv[2] = c_max[0]

    return hsv
