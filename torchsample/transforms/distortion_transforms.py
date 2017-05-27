"""
Transforms to distort local or global information of an image
"""


import torch as th
import numpy as np
import random


class Scramble(object):
    """
    Create blocks of an image and scramble them
    """
    def __init__(self, blocksize):
        self.blocksize = blocksize

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            size = _input.size()
            img_height = size[1]
            img_width = size[2]

            x_blocks = int(img_height/self.blocksize) # number of x blocks
            y_blocks = int(img_width/self.blocksize)
            ind = th.randperm(x_blocks*y_blocks)

            new = th.zeros(_input.size())
            count = 0
            for i in range(x_blocks):
                for j in range (y_blocks):
                    row = int(ind[count] / x_blocks)
                    column = ind[count] % x_blocks
                    new[:, i*self.blocksize:(i+1)*self.blocksize, j*self.blocksize:(j+1)*self.blocksize] = \
                    _input[:, row*self.blocksize:(row+1)*self.blocksize, column*self.blocksize:(column+1)*self.blocksize]
                    count += 1
            outputs.append(new)
        return outputs if idx > 1 else outputs[0]
 

class RandomChoiceScramble(object):

    def __init__(self, blocksizes):
        self.blocksizes = blocksizes

    def __call__(self, *inputs):
        blocksize = random.choice(self.blocksizes)
        outputs = Scramble(blocksize=blocksize)(*inputs)
        return outputs


def _blur_image(image, H):
    # break image up into its color components
    size = image.shape
    imr = image[0,:,:]
    img = image[1,:,:]
    imb = image[2,:,:]

    # compute Fourier transform and frequqnecy spectrum
    Fim1r = np.fft.fftshift(np.fft.fft2(imr))
    Fim1g  = np.fft.fftshift(np.fft.fft2(img))
    Fim1b  = np.fft.fftshift(np.fft.fft2(imb))
    
    # Apply the lowpass filter to the Fourier spectrum of the image
    filtered_imager = np.multiply(H, Fim1r)
    filtered_imageg = np.multiply(H, Fim1g)
    filtered_imageb = np.multiply(H, Fim1b)
    
    newim = np.zeros(size)

    # convert the result to the spatial domain.
    newim[0,:,:] = np.absolute(np.real(np.fft.ifft2(filtered_imager)))
    newim[1,:,:] = np.absolute(np.real(np.fft.ifft2(filtered_imageg)))
    newim[2,:,:] = np.absolute(np.real(np.fft.ifft2(filtered_imageb)))

    return newim.astype('uint8')

def _butterworth_filter(rows, cols, thresh, order):
    # X and Y matrices with ranges normalised to +/- 0.5
    array1 = np.ones(rows)
    array2 = np.ones(cols)
    array3 = np.arange(1,rows+1)
    array4 = np.arange(1,cols+1)

    x = np.outer(array1, array4)
    y = np.outer(array3, array2)

    x = x - float(cols/2) - 1
    y = y - float(rows/2) - 1

    x = x / cols
    y = y / rows

    radius = np.sqrt(np.square(x) + np.square(y))

    matrix1 = radius/thresh
    matrix2 = np.power(matrix1, 2*order)
    f = np.reciprocal(1 + matrix2)

    return f


class Blur(object):
    """
    Blur an image with a Butterworth filter with a frequency
    cutoff matching local block size
    """
    def __init__(self, threshold, order=5):
        """
        scramble blocksize of 128 => filter threshold of 64
        scramble blocksize of 64 => filter threshold of 32
        scramble blocksize of 32 => filter threshold of 16
        scramble blocksize of 16 => filter threshold of 8
        scramble blocksize of 8 => filter threshold of 4
        """
        self.threshold = threshold
        self.order = order

    def __call__(self, *inputs):
        """
        inputs should have values between 0 and 255
        """
        outputs = []
        for idx, _input in enumerate(inputs):
            rows = _input.size(1)
            cols = _input.size(2)
            fc = self.threshold # threshold
            fs = 128.0 # max frequency
            n  = self.order # filter order
            fc_rad = (fc/fs)*0.5
            H = _butterworth_filter(rows, cols, fc_rad, n)
            _input_blurred = _blur_image(_input.numpy().astype('uint8'), H)
            _input_blurred = th.from_numpy(_input_blurred).float()
            outputs.append(_input_blurred)

        return outputs if idx > 1 else outputs[0]


class RandomChoiceBlur(object):

    def __init__(self, thresholds, order=5):
        """
        thresholds = [64.0, 32.0, 16.0, 8.0, 4.0]
        """
        self.thresholds = thresholds
        self.order = order

    def __call__(self, *inputs):
        threshold = random.choice(self.thresholds)
        outputs = Blur(threshold=threshold, order=self.order)(*inputs)
        return outputs






