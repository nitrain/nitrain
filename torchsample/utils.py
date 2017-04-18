"""
Util functions for torchsample
"""

import pickle
import torch
import numpy as np

def save_transform(file, transform):
    """
    Save a transform object
    """
    with open(file, 'wb') as output_file:
        pickler = pickle.Pickler(output_file, -1)
        pickler.dump(transform)


def load_transform(file):
    """
    Load a transform object
    """
    with open(file, 'rb') as input_file:
        transform = pickle.load(input_file)
    return transform


def th_meshgrid(h, w):
    """
    Generates a meshgrid of shape (h*w, 2)

    Arguments
    ---------
    h : integer
    w : integer
    """
    x = torch.range(0, h-1)
    y = torch.range(0, w-1)
    grid = torch.stack([x.repeat(w,1).t().contiguous().view(-1), y.repeat(h)],1)
    return grid


def th_affine_transform(x, matrix, coords=None):
    """
    Affine transform in pytorch. 
    Only supports nearest neighbor interpolation at the moment.

    Assumes channel axis is 2nd dim and there is no sample dim
    e.g. x.size() = (28,28,3)

    Considerations:
        - coordinates outside original image should default to self.fill_value
        - add option for bilinear interpolation

    >>> x = torch.zeros(20,20,1)
    >>> x[5:15,5:15,:] = 1

    """
    #if not x.is_contiguous():
    #    x = x.contiguous()

    # dimensions of image
    H = x.size(0)
    W = x.size(1)
    C = x.size(2)

    # generate coordinate grid if not given
    # can be passed as arg for speed
    if coords is None:
        coords = th_meshgrid(H, W)

    # make the center coordinate the origin
    coords[:,0] -= (H / 2. + 0.5)
    coords[:,1] -= (W / 2. + 0.5)

    # get affine and bias values
    A = matrix[:2,:2].float()
    b = matrix[:2,2].float()

    # perform coordinate transform
    t_coords = coords.mm(A.t().contiguous()) + b.expand_as(coords)

    # move origin coord back to the center
    t_coords[:,0] += (H / 2. + 0.5)
    t_coords[:,1] += (W / 2. + 0.5)

    # round to nearest neighbor
    t_coords = t_coords.round()

    # convert to integer
    t_coords = t_coords.long()

    # clamp any coords outside the original image
    t_coords[:,0] = torch.clamp(t_coords[:,0], 0, H-1)
    t_coords[:,1] = torch.clamp(t_coords[:,1], 0, W-1)

    # flatten image for easier indexing
    x_flat = x.view(-1, C)

    # flatten coordinates for easier indexing
    t_coords_flat = t_coords[:,0]*W + t_coords[:,1]

    # map new coordinates for each channel in original image
    x_mapped = torch.stack([x_flat[:,i][t_coords_flat].view(H,W) 
                    for i in range(C)], 2)

    return x_mapped


def th_pearsonr(x, y):
    """
    mimics scipy.stats.pearsonr
    """
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


def th_corrcoef(x):
    """
    mimics np.corrcoef
    """
    # calculate covariance matrix of rows
    mean_x = torch.mean(x, 1)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)

    # normalize covariance matrix
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())

    # clamp between -1 and 1
    c = torch.clamp(c, -1.0, 1.0)

    return c


def th_matrixcorr(x, y):
    """
    return a correlation matrix between
    columns of x and columns of y.

    So, if X.size() == (1000,4) and Y.size() == (1000,5),
    then the result will be of size (4,5) with the
    (i,j) value equal to the pearsonr correlation coeff
    between column i in X and column j in Y
    """
    mean_x = torch.mean(x, 0)
    mean_y = torch.mean(y, 0)
    xm = x.sub(mean_x.expand_as(x))
    ym = y.sub(mean_y.expand_as(y))
    r_num = xm.t().mm(ym)
    r_den1 = torch.norm(xm,2,0)
    r_den2 = torch.norm(ym,2,0)
    r_den = r_den1.t().mm(r_den2)
    r_mat = r_num.div(r_den)
    return r_mat

