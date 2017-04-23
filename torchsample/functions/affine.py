
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from ..utils import th_meshgrid, th_gather_nd

def F_affine_2d(x, matrix, center=True):
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
    x_transformed = F_bilinear_interp_2d(x, new_coords)

    return x_transformed


def F_bilinear_interp_2d(input, coords):
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


def F_affine_3d(x, matrix, center=True):
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
    x_transformed = F_bilinear_interp_3d(x, new_coords)

    return x_transformed


def F_bilinear_interp_3d(input, coords):
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


def F_bicubic_interp_2d(input, coords):
    def cubic_hermite(A, B, C, D, t):
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

