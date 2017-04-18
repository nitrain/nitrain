"""
Util functions for torchsample
"""

import pickle
import torch

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


def th_random_choice(a, size=None, replace=True, p=None):
    """
    Parameters
    -----------
    a : 1-D array-like
        If a torch.Tensor, a random sample is generated from its elements.
        If an int, the random sample is generated as if a was torch.range(n)
    size : int, optional
        Number of samples to draw. Default is None, in which case a
        single value is returned.
    replace : boolean, optional
        Whether the sample is with or without replacement
    p : 1-D array-like, optional
        The probabilities associated with each entry in a.
        If not given the sample assumes a uniform distribution over all
        entries in a.

    Returns
    --------
    samples : 1-D ndarray, shape (size,)
        The generated random samples
    
    Notes
    -----
    Handle:
        - a is 1D tensor, size is None

    Example
    -------
    - with size = 100,000:
        - no probabilities
            >>> x = torch.range(0,4)
            >>> xx = th_random_choice(x,size=100000))
            >>> print('%i - %.03f' % (0, torch.sum(xx==0)/100000))
            >>> print('%i - %.03f' % (1, torch.sum(xx==1)/100000))
            >>> print('%i - %.03f' % (2, torch.sum(xx==2)/100000))
            >>> print('%i - %.03f' % (3, torch.sum(xx==3)/100000))
            >>> print('%i - %.03f' % (4, torch.sum(xx==4)/100000))
            >>> print('\n')

        - probabilities
            >>> x = torch.range(0,4)
            >>> xx = th_random_choice(x,size=100000, p=[0.1,0.2,0.3,0.05,0.35])
            >>> print('%.03f - %.03f' % (0.1, torch.sum(xx==0)/100000))
            >>> print('%.03f - %.03f' % (0.2, torch.sum(xx==1)/100000))
            >>> print('%.03f - %.03f' % (0.3, torch.sum(xx==2)/100000))
            >>> print('%.03f - %.03f' % (0.05, torch.sum(xx==3)/100000))
            >>> print('%.03f - %.03f' % (0.35, torch.sum(xx==4)/100000))
            >>> print('\n')
    """
    if size is None:
        size = 1

    if isinstance(a, int):
        a = torch.range(0, a-1)

    if p is None:
        if replace:
            idx = torch.floor(torch.rand(size)*a.size(0)).long()
        else:
            idx = torch.randperm(a.size(0))[:size]
    else:
        if abs(1.0-sum(p)) > 1e-3:
            raise ValueError('p must sum to 1.0')
        if not replace:
            raise ValueError('replace must equal true if probabilities given')
        idx_vec = torch.cat([torch.zeros(round(p[i]*1000))+i for i in range(len(p))])
        idx = (torch.floor(torch.rand(size)*999.99)).long()
        idx = idx_vec[idx].long()
    return a[idx]


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

