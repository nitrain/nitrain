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
    dtype = torch.LongTensor
    pools = []
    for i in args:
        if isinstance(i, int):
            pools.append(torch.range(0, i-1))
        else:
            if type(i) != torch.LongTensor:
                print
                dtype = i.type()
            pools.append(i)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    return torch.Tensor(result).type(dtype)


def th_meshgrid_like(x):
    return th_meshgrid(*x.size())


def th_gather_nd(x, coords):
    inds = coords.mv(torch.LongTensor(x.stride()))
    x_gather = torch.index_select(th_flatten(x), 0, inds)
    return x_gather


def th_affine_2d(x, matrix, mode='bilinear', center=True):
    """
    2D Affine image transform on torch.Tensor

    Arguments
    ---------
    x : torch.Tensor of size (C, H, W)
        image tensor to be transformed

    matrix : torch.Tensor of size (3, 3) or (2, 3)
        transformation matrix

    mode : string in {'nearest', 'bilinear'}
        interpolation scheme to use

    center : boolean
        whether to alter the bias of the transform 
        so the transform is applied about the center
        of the image rather than the origin

    Example
    -------
    >>> x = torch.zeros(3,20,20)
    >>> x[:2,5:15,5:15] = 1
    >>> x[-1,2:12,7:17] = 1
    >>> matrix = torch.FloatTensor([[1.,0,-5],[0,1.,0]])
    >>> xn = th_affine_2d(x, matrix, mode='nearest')
    >>> xb = th_affine_2d(x, matrix, mode='bilinear')
    """
    # grab A and b weights from 2x3 matrix
    A = matrix[:2,:2]
    b = matrix[:2,2]

    # make a meshgrid of normal coordinates
    coords = th_meshgrid(x.size(1),x.size(2)).float()

    if center:
        # shift the coordinates so center is the origin
        coords[:,0] = coords[:,0] - (x.size(1) / 2. + 0.5)
        coords[:,1] = coords[:,1] - (x.size(2) / 2. + 0.5)

    # apply the coordinate transformation
    new_coords = coords.mm(A.t().contiguous()) + b.expand_as(coords)

    if center:
        # shift the coordinates back so origin is origin
        new_coords[:,0] = new_coords[:,0] + (x.size(1) / 2. + 0.5)
        new_coords[:,1] = new_coords[:,1] + (x.size(2) / 2. + 0.5)

    # map new coordinates using bilinear interpolation
    if mode == 'nearest':
        x_transformed = th_nearest_interp_2d(x, new_coords)
    elif mode == 'bilinear':
        x_transformed = th_bilinear_interp_2d(x, new_coords)

    return x_transformed


def th_nearest_interp_2d(input, coords):
    """
    2d nearest neighbor interpolation torch.Tensor
    """
    xc = torch.clamp(coords[:,0], 0, input.size(1)-1)
    yc = torch.clamp(coords[:,1], 0, input.size(2)-1)

    coords = torch.stack([xc.round().long(), 
                          yc.round().long()], 1)

    mapped_vals = torch.stack([th_gather_nd(input[i], coords)
                    for i in range(input.size(0))], 0)

    return mapped_vals.view_as(input)


def th_bilinear_interp_2d(input, coords):
    """
    bilinear interpolation of 2d torch.autograd.Variable

    Arguments
    ---------
    input : torch.FloatTensor of size (C, H, W)
        image to interpolate

    coords : torch.FloatTensor of size (C*H*W, 2)
        coordinates to index the input on
    """
    xc = torch.clamp(coords[:,0], 0, input.size(1)-1)
    yc = torch.clamp(coords[:,1], 0, input.size(2)-1)
    coords = torch.stack([xc, yc],1)

    coords_lt = coords.floor().long()
    coords_rb = coords.ceil().long()
    coords_lb = torch.stack([coords_lt[:, 0], coords_rb[:, 1]], 1)
    coords_rt = torch.stack([coords_rb[:, 0], coords_lt[:, 1]], 1)

    vals_lt = torch.stack([th_gather_nd(input[i], coords_lt)
                    for i in range(input.size(0))],0)
    vals_rb = torch.stack([th_gather_nd(input[i], coords_rb)
                    for i in range(input.size(0))],0)
    vals_lb = torch.stack([th_gather_nd(input[i], coords_lb)
                    for i in range(input.size(0))],0)
    vals_rt = torch.stack([th_gather_nd(input[i], coords_rt)
                    for i in range(input.size(0))],0)
    
    c_u = coords.unsqueeze(0).repeat(input.size(0),1,1)
    clt_u = coords_lt.unsqueeze(0).repeat(input.size(0),1,1).type(coords.type())
    coords_offset_lt = c_u - clt_u
                       
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:,:,0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:,:,0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:,:,1]

    return mapped_vals.view_as(input)


def th_affine_3d(x, matrix, mode='bilinear', center=True):
    """
    3D Affine image transform on torch.Tensor

    Arguments
    ---------
    x : torch.Tensor of size (C, D, H, W)
        image tensor to be transformed

    matrix : torch.Tensor of size (3, 4) or (4, 4)
        transformation matrix

    mode : string in {'nearest', 'bilinear'}
        interpolation scheme to use

    center : boolean
        whether to alter the bias of the transform 
        so the transform is applied about the center
        of the image rather than the origin

    Example
    -------
    >>> x = torch.zeros(1,20,20,20)
    >>> x[:,5:15,5:15,5:15] = 1
    >>> matrix = torch.FloatTensor([[1.2, 0, 0, 0],
                                    [0, 1.2, 0, 0],
                                    [0, 0, 1.2, 0]])
    >>> xn = th_affine_3d(x, matrix, mode='nearest')
    >>> xb = th_affine_3d(x, matrix, mode='bilinear')

    """
    # grab A and b weights from 2x3 matrix
    A = matrix[:3,:3]
    b = matrix[:3,3]

    # make a meshgrid of normal coordinates
    coords = th_meshgrid(x.size(1),x.size(2),x.size(3)).float()

    if center:
        # shift the coordinates so center is the origin
        coords[:,0] = coords[:,0] - (x.size(1) / 2. + 0.5)
        coords[:,1] = coords[:,1] - (x.size(2) / 2. + 0.5)
        coords[:,2] = coords[:,2] - (x.size(3) / 2. + 0.5)

    # apply the coordinate transformation
    new_coords = coords.mm(A.t().contiguous()) + b.expand_as(coords)

    if center:
        # shift the coordinates back so origin is origin
        new_coords[:,0] = new_coords[:,0] + (x.size(1) / 2. + 0.5)
        new_coords[:,1] = new_coords[:,1] + (x.size(2) / 2. + 0.5)
        new_coords[:,2] = new_coords[:,2] + (x.size(3) / 2. + 0.5)

    # map new coordinates using bilinear interpolation
    if mode == 'nearest':
        x_transformed = th_nearest_interp_3d(x, new_coords)
    elif mode == 'bilinear':
        x_transformed = th_bilinear_interp_3d(x, new_coords)

    return x_transformed


def th_nearest_interp_3d(input, coords):
    """
    2d nearest neighbor interpolation torch.Tensor
    """
    xc = torch.clamp(coords[:,0], 0, input.size(1)-1)
    yc = torch.clamp(coords[:,1], 0, input.size(2)-1)
    zc = torch.clamp(coords[:,2], 0, input.size(3)-1)

    coords = torch.stack([xc.round().long(),
                          yc.round().long(),
                          zc.round().long()], 1)

    mapped_vals = torch.stack([th_gather_nd(input[i], coords)
                    for i in range(input.size(0))], 0)

    return mapped_vals.view_as(input)


def th_bilinear_interp_3d(input, coords):
    """
    trilinear interpolation of 3D image
    """
    x = torch.clamp(coords[:,0], 0, input.size(1)-1.00001)
    x0 = x.floor().long()
    x1 = x0 + 1

    y = torch.clamp(coords[:,1], 0, input.size(2)-1.00001)
    y0 = y.floor().long()
    y1 = y0 + 1

    z = torch.clamp(coords[:,2], 0, input.size(3)-1.00001)
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

    vals_000 = th_gather_nd(input, c_000)
    vals_111 = th_gather_nd(input, c_111)
    vals_001 = th_gather_nd(input, c_001)
    vals_010 = th_gather_nd(input, c_010)
    vals_011 = th_gather_nd(input, c_011)
    vals_100 = th_gather_nd(input, c_100)
    vals_110 = th_gather_nd(input, c_110)
    vals_101 = th_gather_nd(input, c_101)

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
    


