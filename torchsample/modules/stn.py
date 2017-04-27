
import torch.nn as nn

from ..functions import F_affine2d, F_affine3d


class STN2d(nn.Module):

    def __init__(self, local_net):
        super(STN2d, self).__init__()
        self.local_net = local_net

    def forward(self, x):
        params = self.local_net(x)
        x_transformed = F_affine2d(x[0], params.view(2,3))
        return x_transformed


class STN3d(nn.Module):

    def __init__(self, local_net):
        self.local_net = local_net

    def forward(self, x):
        params = self.local_net(x)
        x_transformed = F_affine3d(x, params.view(3,4))
        return x_transformed

