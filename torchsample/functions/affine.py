
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from ..utils import th_iterproduct, th_flatten


def F_affine2d(x, matrix, center=True):
    """
    2D Affine image transform on torch.autograd.Variable
    """
    if matrix.dim() == 2:
        matrix = matrix.view(-1,2,3)

    A_batch = matrix[:,:,:2]
    if A_batch.size(0) != x.size(0):
        A_batch = A_batch.repeat(x.size(0),1,1)
    b_batch = matrix[:,:,2].unsqueeze(1)

    # make a meshgrid of normal coordinates
    _coords = th_iterproduct(x.size(1),x.size(2))
    coords = Variable(_coords.unsqueeze(0).repeat(x.size(0),1,1).float(),
                    requires_grad=False)
    if center:
        # shift the coordinates so center is the origin
        coords[:,:,0] = coords[:,:,0] - (x.size(1) / 2. + 0.5)
        coords[:,:,1] = coords[:,:,1] - (x.size(2) / 2. + 0.5)

    # apply the coordinate transformation
    new_coords = coords.bmm(A_batch.transpose(1,2)) + b_batch.expand_as(coords)

    if center:
        # shift the coordinates back so origin is origin
        new_coords[:,:,0] = new_coords[:,:,0] + (x.size(1) / 2. + 0.5)
        new_coords[:,:,1] = new_coords[:,:,1] + (x.size(2) / 2. + 0.5)

    # map new coordinates using bilinear interpolation
    x_transformed = F_bilinear_interp2d(x, new_coords)

    return x_transformed


def F_bilinear_interp2d(input, coords):
    """
    bilinear interpolation of 2d torch.autograd.Variable
    """
    x = torch.clamp(coords[:,:,0], 0, input.size(1)-2)
    x0 = x.floor()
    x1 = x0 + 1
    y = torch.clamp(coords[:,:,1], 0, input.size(2)-2)
    y0 = y.floor()
    y1 = y0 + 1

    stride = torch.LongTensor(input.stride())
    x0_ix = x0.mul(stride[1]).long()
    x1_ix = x1.mul(stride[1]).long()
    y0_ix = y0.mul(stride[2]).long()
    y1_ix = y1.mul(stride[2]).long()

    input_flat = input.view(input.size(0),-1).contiguous()

    vals_00 = input_flat.gather(1, x0_ix.add(y0_ix).detach())
    vals_10 = input_flat.gather(1, x1_ix.add(y0_ix).detach())
    vals_01 = input_flat.gather(1, x0_ix.add(y1_ix).detach())
    vals_11 = input_flat.gather(1, x1_ix.add(y1_ix).detach())
    
    xd = x - x0
    yd = y - y0
    xm = 1 - xd
    ym = 1 - yd

    x_mapped = (vals_00.mul(xm).mul(ym) +
                vals_10.mul(xd).mul(ym) +
                vals_01.mul(xm).mul(yd) +
                vals_11.mul(xd).mul(yd))

    return x_mapped.view_as(input)


def F_batch_affine2d(x, matrix, center=True):
    """

    x : torch.Tensor
        shape = (Samples, C, H, W)
        NOTE: Assume C is always equal to 1!
    matrix : torch.Tensor
        shape = (Samples, 6) or (Samples, 2, 3)

    Example
    -------
    >>> x = Variable(torch.zeros(3,1,10,10))
    >>> x[:,:,3:7,3:7] = 1
    >>> m1 = torch.FloatTensor([[1.2,0,0],[0,1.2,0]])
    >>> m2 = torch.FloatTensor([[0.8,0,0],[0,0.8,0]])
    >>> m3 = torch.FloatTensor([[1.0,0,3],[0,1.0,3]])
    >>> matrix = Variable(torch.stack([m1,m2,m3]))
    >>> xx = F_batch_affine2d(x,matrix)
    """
    if matrix.dim() == 2:
        matrix = matrix.view(-1,2,3)

    A_batch = matrix[:,:,:2]
    b_batch = matrix[:,:,2].unsqueeze(1)

    # make a meshgrid of normal coordinates
    _coords = th_iterproduct(x.size(2),x.size(3))
    coords = Variable(_coords.unsqueeze(0).repeat(x.size(0),1,1).float(),
                requires_grad=False)

    if center:
        # shift the coordinates so center is the origin
        coords[:,:,0] = coords[:,:,0] - (x.size(2) / 2. + 0.5)
        coords[:,:,1] = coords[:,:,1] - (x.size(3) / 2. + 0.5)
    
    # apply the coordinate transformation
    new_coords = coords.bmm(A_batch.transpose(1,2)) + b_batch.expand_as(coords)

    if center:
        # shift the coordinates back so origin is origin
        new_coords[:,:,0] = new_coords[:,:,0] + (x.size(2) / 2. + 0.5)
        new_coords[:,:,1] = new_coords[:,:,1] + (x.size(3) / 2. + 0.5)

    # map new coordinates using bilinear interpolation
    x_transformed = F_batch_bilinear_interp2d(x, new_coords)

    return x_transformed


def F_batch_bilinear_interp2d(input, coords):
    """
    input : torch.Tensor
        size = (N,H,W,C)
    coords : torch.Tensor
        size = (N,H*W*C,2)
    """
    x = torch.clamp(coords[:,:,0], 0, input.size(2)-2)
    x0 = x.floor()
    x1 = x0 + 1
    y = torch.clamp(coords[:,:,1], 0, input.size(3)-2)
    y0 = y.floor()
    y1 = y0 + 1

    stride = torch.LongTensor(input.stride())
    x0_ix = x0.mul(stride[2]).long()
    x1_ix = x1.mul(stride[2]).long()
    y0_ix = y0.mul(stride[3]).long()
    y1_ix = y1.mul(stride[3]).long()

    input_flat = input.view(input.size(0),-1).contiguous()

    vals_00 = input_flat.gather(1, x0_ix.add(y0_ix).detach())
    vals_10 = input_flat.gather(1, x1_ix.add(y0_ix).detach())
    vals_01 = input_flat.gather(1, x0_ix.add(y1_ix).detach())
    vals_11 = input_flat.gather(1, x1_ix.add(y1_ix).detach())
    
    xd = x - x0
    yd = y - y0
    xm = 1 - xd
    ym = 1 - yd

    x_mapped = (vals_00.mul(xm).mul(ym) +
                vals_10.mul(xd).mul(ym) +
                vals_01.mul(xm).mul(yd) +
                vals_11.mul(xd).mul(yd))

    return x_mapped.view_as(input)


def F_affine3d(x, matrix, center=True):
    A = matrix[:3,:3]
    b = matrix[:3,3]

    # make a meshgrid of normal coordinates
    coords = Variable(th_iterproduct(x.size(1),x.size(2),x.size(3)).float(),
                requires_grad=False)

    if center:
        # shift the coordinates so center is the origin
        coords[:,0] = coords[:,0] - (x.size(1) / 2. + 0.5)
        coords[:,1] = coords[:,1] - (x.size(2) / 2. + 0.5)
        coords[:,2] = coords[:,2] - (x.size(3) / 2. + 0.5)

    
    # apply the coordinate transformation
    new_coords = F.linear(coords, A, b)

    if center:
        # shift the coordinates back so origin is origin
        new_coords[:,0] = new_coords[:,0] + (x.size(1) / 2. + 0.5)
        new_coords[:,1] = new_coords[:,1] + (x.size(2) / 2. + 0.5)
        new_coords[:,2] = new_coords[:,2] + (x.size(3) / 2. + 0.5)

    # map new coordinates using bilinear interpolation
    x_transformed = F_trilinear_interp3d(x, new_coords)

    return x_transformed


def F_trilinear_interp3d(input, coords):
    """
    trilinear interpolation of 3D image
    """
    # take clamp then floor/ceil of x coords
    x = torch.clamp(coords[:,0], 0, input.size(1)-2)
    x0 = x.floor()
    x1 = x0 + 1
    # take clamp then floor/ceil of y coords
    y = torch.clamp(coords[:,1], 0, input.size(2)-2)
    y0 = y.floor()
    y1 = y0 + 1
    # take clamp then floor/ceil of z coords
    z = torch.clamp(coords[:,2], 0, input.size(3)-2)
    z0 = z.floor()
    z1 = z0 + 1

    stride = torch.LongTensor(input.stride())[1:]
    x0_ix = x0.mul(stride[0]).long()
    x1_ix = x1.mul(stride[0]).long()
    y0_ix = y0.mul(stride[1]).long()
    y1_ix = y1.mul(stride[1]).long()
    z0_ix = z0.mul(stride[2]).long()
    z1_ix = z1.mul(stride[2]).long()

    input_flat = th_flatten(input)

    vals_000 = input_flat[x0_ix.add(y0_ix).add(z0_ix).detach()]
    vals_100 = input_flat[x1_ix.add(y0_ix).add(z0_ix).detach()]
    vals_010 = input_flat[x0_ix.add(y1_ix).add(z0_ix).detach()]
    vals_001 = input_flat[x0_ix.add(y0_ix).add(z1_ix).detach()]
    vals_101 = input_flat[x1_ix.add(y0_ix).add(z1_ix).detach()]
    vals_011 = input_flat[x0_ix.add(y1_ix).add(z1_ix).detach()]
    vals_110 = input_flat[x1_ix.add(y1_ix).add(z0_ix).detach()]
    vals_111 = input_flat[x1_ix.add(y1_ix).add(z1_ix).detach()]

    xd = x - x0
    yd = y - y0
    zd = z - z0
    xm = 1 - xd
    ym = 1 - yd
    zm = 1 - zd

    x_mapped = (vals_000.mul(xm).mul(ym).mul(zm) +
                vals_100.mul(xd).mul(ym).mul(zm) +
                vals_010.mul(xm).mul(yd).mul(zm) +
                vals_001.mul(xm).mul(ym).mul(zd) +
                vals_101.mul(xd).mul(ym).mul(zd) +
                vals_011.mul(xm).mul(yd).mul(zd) +
                vals_110.mul(xd).mul(yd).mul(zm) +
                vals_111.mul(xd).mul(yd).mul(zd))

    return x_mapped.view_as(input)


def F_batch_affine3d(x, matrix, center=True):
    """

    x : torch.Tensor
        shape = (Samples, C, H, W)
        NOTE: Assume C is always equal to 1!
    matrix : torch.Tensor
        shape = (Samples, 6) or (Samples, 2, 3)

    Example
    -------
    >>> x = Variable(torch.zeros(3,1,10,10,10))
    >>> x[:,:,3:7,3:7,3:7] = 1
    >>> m1 = torch.FloatTensor([[1.2,0,0,0],[0,1.2,0,0],[0,0,1.2,0]])
    >>> m2 = torch.FloatTensor([[0.8,0,0,0],[0,0.8,0,0],[0,0,0.8,0]])
    >>> m3 = torch.FloatTensor([[1.0,0,0,3],[0,1.0,0,3],[0,0,1.0,3]])
    >>> matrix = Variable(torch.stack([m1,m2,m3]))
    >>> xx = F_batch_affine3d(x,matrix)
    """
    if matrix.dim() == 2:
        matrix = matrix.view(-1,3,4)

    A_batch = matrix[:,:3,:3]
    b_batch = matrix[:,:3,3].unsqueeze(1)

    # make a meshgrid of normal coordinates
    _coords = th_iterproduct(x.size(2),x.size(3),x.size(4))
    coords = Variable(_coords.unsqueeze(0).repeat(x.size(0),1,1).float(),
                requires_grad=False)
    
    if center:
        # shift the coordinates so center is the origin
        coords[:,:,0] = coords[:,:,0] - (x.size(2) / 2. + 0.5)
        coords[:,:,1] = coords[:,:,1] - (x.size(3) / 2. + 0.5)
        coords[:,:,2] = coords[:,:,2] - (x.size(4) / 2. + 0.5)
    
    # apply the coordinate transformation
    new_coords = coords.bmm(A_batch.transpose(1,2)) + b_batch.expand_as(coords)

    if center:
        # shift the coordinates back so origin is origin
        new_coords[:,:,0] = new_coords[:,:,0] + (x.size(2) / 2. + 0.5)
        new_coords[:,:,1] = new_coords[:,:,1] + (x.size(3) / 2. + 0.5)
        new_coords[:,:,2] = new_coords[:,:,2] + (x.size(4) / 2. + 0.5)

    # map new coordinates using bilinear interpolation
    x_transformed = F_batch_trilinear_interp3d(x, new_coords)

    return x_transformed


def F_batch_trilinear_interp3d(input, coords):
    """
    input : torch.Tensor
        size = (N,H,W,C)
    coords : torch.Tensor
        size = (N,H*W*C,2)
    """
    x = torch.clamp(coords[:,:,0], 0, input.size(2)-2)
    x0 = x.floor()
    x1 = x0 + 1
    y = torch.clamp(coords[:,:,1], 0, input.size(3)-2)
    y0 = y.floor()
    y1 = y0 + 1
    z = torch.clamp(coords[:,:,2], 0, input.size(4)-2)
    z0 = z.floor()
    z1 = z0 + 1

    stride = torch.LongTensor(input.stride())
    x0_ix = x0.mul(stride[2]).long()
    x1_ix = x1.mul(stride[2]).long()
    y0_ix = y0.mul(stride[3]).long()
    y1_ix = y1.mul(stride[3]).long()
    z0_ix = z0.mul(stride[4]).long()
    z1_ix = z1.mul(stride[4]).long()

    input_flat = input.contiguous().view(input.size(0),-1)

    vals_000 = input_flat.gather(1,x0_ix.add(y0_ix).add(z0_ix).detach())
    vals_100 = input_flat.gather(1,x1_ix.add(y0_ix).add(z0_ix).detach())
    vals_010 = input_flat.gather(1,x0_ix.add(y1_ix).add(z0_ix).detach())
    vals_001 = input_flat.gather(1,x0_ix.add(y0_ix).add(z1_ix).detach())
    vals_101 = input_flat.gather(1,x1_ix.add(y0_ix).add(z1_ix).detach())
    vals_011 = input_flat.gather(1,x0_ix.add(y1_ix).add(z1_ix).detach())
    vals_110 = input_flat.gather(1,x1_ix.add(y1_ix).add(z0_ix).detach())
    vals_111 = input_flat.gather(1,x1_ix.add(y1_ix).add(z1_ix).detach())

    xd = x - x0
    yd = y - y0
    zd = z - z0
    xm = 1 - xd
    ym = 1 - yd
    zm = 1 - zd

    x_mapped = (vals_000.mul(xm).mul(ym).mul(zm) +
                vals_100.mul(xd).mul(ym).mul(zm) +
                vals_010.mul(xm).mul(yd).mul(zm) +
                vals_001.mul(xm).mul(ym).mul(zd) +
                vals_101.mul(xd).mul(ym).mul(zd) +
                vals_011.mul(xm).mul(yd).mul(zd) +
                vals_110.mul(xd).mul(yd).mul(zm) +
                vals_111.mul(xd).mul(yd).mul(zd))

    return x_mapped.view_as(input)


