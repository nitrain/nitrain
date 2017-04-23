"""
Utility functions for torch.Tensors
"""

import pickle
import torch


def th_allclose(x, y):
    """
    Determine whether two torch tensors have same values
    Mimics np.allclose
    """
    return torch.sum(torch.abs(x-y)) < 1e-5


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


def th_meshgrid(*args):
    pools = (torch.range(0,i-1) for i in args)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return torch.Tensor(result).long()


def th_affine2d(x, matrix, coords=None):
    """
    Affine image transform on torch.Tensor

    Only supports nearest neighbor interpolation at the moment.

    Assumes channel axis is 0th dim and there is no sample dim
    e.g. x.size() = (1,28,28)

    Considerations:
        - coordinates outside original image should default to self.fill_value
        - add option for bilinear interpolation

    >>> x = torch.zeros(20,20,1)
    >>> x[5:15,5:15,:] = 1

    """
    #if not x.is_contiguous():
    #    x = x.contiguous()

    # dimensions of image
    C = x.size(0)
    H = x.size(1)
    W = x.size(2)
    
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

    # convert to long
    t_coords = t_coords.long()

    # clamp any coords outside the original image
    t_coords[:,0] = torch.clamp(t_coords[:,0], 0, H-1)
    t_coords[:,1] = torch.clamp(t_coords[:,1], 0, W-1)

    # flatten image for easier indexing
    x_flat = x.view(C, -1)

    # flatten coordinates for easier indexing
    t_coords_flat = t_coords[:,0]*W + t_coords[:,1]

    # map new coordinates for each channel in original image
    x_mapped = torch.stack([x_flat[i][t_coords_flat].view(H,W) 
                    for i in range(C)], 0)

    return x_mapped



def th_gather_nd(x, coords):
    """
    Returns a flattened tensor of x indexed by coords

    Example:
        >>> x = torch.randn(2,3,1) # random 3d tensor
        >>> coords = th_meshgrid(2,3,1) # create coordinate grid
        >>> xx = th_gather_nd(x, coords).view_as(x) # gather and view
        >>> print(th_allclose(x, xx)) # True
    """
    if coords.size(1) != x.dim():
        raise ValueError('Coords must have column for each image dim')

    inds = coords[:,0]*x.size(1)
    for i in range(x.dim()-2):
        inds += coords[:,i+1]*x.size(i+2)
    inds += coords[:,-1]
    x_gather = torch.index_select(th_flatten(x), 0, inds)
    return x_gather


def th_flatten(x):
    """Flatten tensor"""
    return x.contiguous().view(-1)


def th_c_flatten(x):
    """
    Flatten tensor, leaving channel intact.
    Assumes CHW format.
    """
    return x.contiguous().view(x.size(0), -1)


def th_bc_flatten(x):
    """
    Flatten tensor, leaving batch and channel dims intact.
    Assumes BCHW format
    """
    return x.contiguous().view(x.size(0), x.size(1), -1)


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
    


