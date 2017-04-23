"""
Util functions for torchsample
"""

import pickle
import torch

from torch.autograd import Variable
import torch.nn.functional as F

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


def F_affine2d(x, matrix, center=True):
    """
    2D Affine image transform on torch.autograd.Variable
    """
    # grab A and b weights from 2x3 matrix
    A = matrix[:2,:2]
    b = matrix[:2,2]

    # make a meshgrid of normal coordinates
    coords = Variable(th_meshgrid(x.size(0),x.size(1)), requires_grad=False)

    if center:
        # shift the coordinates so center is the origin
        coords[:,0] = coords[:,0] - (x.size(0) / 2. + 0.5)
        coords[:,1] = coords[:,1] - (x.size(1) / 2. + 0.5)

    # apply the coordinate transformation
    new_coords = F.linear(coords, A, b)

    if center:
        # shift the coordinates back so origin is origin
        new_coords[:,0] = new_coords[:,0] + (x.size(0) / 2. + 0.5)
        new_coords[:,1] = new_coords[:,1] + (x.size(1) / 2. + 0.5)

    # map new coordinates using bilinear interpolation
    x_transformed = F_map_coordinates2d(x, new_coords)

    return x_transformed


def F_map_coordinates2d(input, coords):
    """
    bilinear interpolation of 2d torch.autograd.Variable
    """
    xc = torch.clamp(coords[:,0], 0, input.size(0)-1)
    yc = torch.clamp(coords[:,1], 0, input.size(1)-1)
    coords = torch.stack([xc,yc],1)

    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    coords_lb = torch.stack([coords_lt[:, 0], coords_rb[:, 1]], 1)
    coords_rt = torch.stack([coords_rb[:, 0], coords_lt[:, 1]], 1)

    vals_lt = th_gather_nd(input,  coords_lt.detach())
    vals_rb = th_gather_nd(input,  coords_rb.detach())
    vals_lb = th_gather_nd(input,  coords_lb.detach())
    vals_rt = th_gather_nd(input,  coords_rt.detach())

    coords_offset_lt = coords - coords_lt.type(coords.data.type())

    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, 1]
    return mapped_vals.view_as(input)


def F_affine3d(x, matrix, center=True):
    # grab A and b weights from 2x3 matrix
    A = matrix[:3,:3]
    b = matrix[:3,3]

    # make a meshgrid of normal coordinates
    coords = Variable(th_meshgrid(x.size(0),x.size(1),x.size(2)), 
                      requires_grad=False)

    if center:
        # shift the coordinates so center is the origin
        coords[:,0] = coords[:,0] - (x.size(0) / 2. + 0.5)
        coords[:,1] = coords[:,1] - (x.size(1) / 2. + 0.5)
        coords[:,2] = coords[:,2] - (x.size(2) / 2. + 0.5)

    # apply the coordinate transformation
    new_coords = F.linear(coords, A, b)

    if center:
        # shift the coordinates back so origin is origin
        new_coords[:,0] = new_coords[:,0] + (x.size(0) / 2. + 0.5)
        new_coords[:,1] = new_coords[:,1] + (x.size(1) / 2. + 0.5)
        new_coords[:,2] = new_coords[:,2] + (x.size(2) / 2. + 0.5)

    # map new coordinates using bilinear interpolation
    x_transformed = F_map_coordinates3d(x, new_coords)

    return x_transformed


def F_map_coordinates3d(input, coords):
    """
    trilinear interpolation of 3D image
    """
    x = torch.clamp(coords[:,0], 0, input.size(0)-1.00001)
    x0 = x.floor().long()
    x1 = x0 + 1

    y = torch.clamp(coords[:,1], 0, input.size(1)-1.00001)
    y0 = y.floor().long()
    y1 = y0 + 1

    z = torch.clamp(coords[:,2], 0, input.size(2)-1.00001)
    z0 = z.floor().long()
    z1 = z0 + 1

    c_000 = torch.stack([x0,y0,z0])
    c_111 = torch.stack([x1,y1,z1])
    c_001 = torch.stack([x0,y0,z1])
    c_010 = torch.stack([x0,y1,z0])
    c_011 = torch.stack([x0,y1,z1])
    c_100 = torch.stack([x1,y0,z0])
    c_110 = torch.stack([x1,y1,z0])
    c_101 = torch.stack([x1,y0,z1])

    vals_000 = th_gather_nd(input, c_000.detach())
    vals_111 = th_gather_nd(input, c_111.detach())
    vals_001 = th_gather_nd(input, c_001.detach())
    vals_010 = th_gather_nd(input, c_010.detach())
    vals_011 = th_gather_nd(input, c_011.detach())
    vals_100 = th_gather_nd(input, c_100.detach())
    vals_110 = th_gather_nd(input, c_110.detach())
    vals_101 = th_gather_nd(input, c_101.detach())

    xd = ((x-x0)/(x1-x0))
    yd = (y-y0)/(y1-y0)
    zd = (z-z0)/(z1-z0)

    c00 = vals_000*(1-xd) + vals_100*xd
    c01 = vals_001*(1-xd) + vals_101*xd
    c10 = vals_010*(1-xd) + vals_110*xd
    c11 = vals_011*(1-xd) + vals_111*xd

    c0 = c00*(1-yd) + c10*yd
    c1 = c01*(1-yd) + c11*yd

    c = c0*(1-zd) + c1*zd

    return c.view_as(input)


def F_map_coordinates2d_2(input, coords):
    x = torch.clamp(coords[:,0], 0, input.size(0)-1.00001)
    x0 = x.floor().long()
    x1 = x0 + 1

    y = torch.clamp(coords[:,1], 0, input.size(1)-1.00001)
    y0 = y.floor().long()
    y1 = y0 + 1

    c_00 = torch.stack([x0,y0])
    c_11 = torch.stack([x1,y1])
    c_01 = torch.stack([x0,y1])
    c_10 = torch.stack([x1,y0])

    vals_00 = th_gather_nd(input, c_00.detach())
    vals_11 = th_gather_nd(input, c_11.detach())
    vals_01 = th_gather_nd(input, c_01.detach())
    vals_10 = th_gather_nd(input, c_10.detach())

    c0 = ((x1-x)/(x1-x0))*vals_00 + ((x-x1)/(x1-x0))*vals_10
    c1 = ((x1-x)/(x1-x0))*vals_01 + ((x-x0)/(x1-x0))*vals_11
    c = ((y1-y)/(y1-y0))*c0 + ((y-y0)/(y1-y0))*c1

    return c.view_as(input)


def F_bicubic2d(input, coords):
    def cubic_hermite(A,B,C,D,t):
        a = -A / 2.0 + (3.0*B) / 2.0 - (3.0*C) / 2.0 + D / 2.0
        b = A - (5.0*B) / 2.0 + 2.0*C - D / 2.0
        c = -A / 2.0 + C / 2.0
        d = B
        return a*t*t*t + b*t*t + c*t + d

    u = coords[:,0]
    v = coords[:,0]

    x = u - 0.5
    xint = x.long()
    xfract = x - x.floor()

    y = v - 0.5
    yint = y.long()
    yfract = y - y.floor()

    p00 = th_gather_nd(input, torch.stack([xint - 1, yint - 1]))
    p10 = th_gather_nd(input, torch.stack([xint + 0, yint - 1]))
    p20 = th_gather_nd(input, torch.stack([xint + 1, yint - 1]))
    p30 = th_gather_nd(input, torch.stack([xint + 2, yint - 1]))

    p01 = th_gather_nd(input, torch.stack([xint - 1, yint + 0]))
    p11 = th_gather_nd(input, torch.stack([xint + 0, yint + 0]))
    p21 = th_gather_nd(input, torch.stack([xint + 1, yint + 0]))
    p31 = th_gather_nd(input, torch.stack([xint + 2, yint + 0]))
 
    p02 = th_gather_nd(input, torch.stack([xint - 1, yint + 1]))
    p12 = th_gather_nd(input, torch.stack([xint + 0, yint + 1]))
    p22 = th_gather_nd(input, torch.stack([xint + 1, yint + 1]))
    p32 = th_gather_nd(input, torch.stack([xint + 2, yint + 1]))
 
    p03 = th_gather_nd(input, torch.stack([xint - 1, yint + 2]))
    p13 = th_gather_nd(input, torch.stack([xint + 0, yint + 2]))
    p23 = th_gather_nd(input, torch.stack([xint + 1, yint + 2]))
    p33 = th_gather_nd(input, torch.stack([xint + 2, yint + 2]))

    col0 = cubic_hermite(p00,p10,p20,p30,xfract)
    col1 = cubic_hermite(p01,p11,p21,p31,xfract)
    col2 = cubic_hermite(p02,p12,p22,p32,xfract)
    col3 = cubic_hermite(p03,p13,p23,p33,xfract)

    x_mapped = cubic_hermite(col0,col1,col2,col3,yfract)
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
    


