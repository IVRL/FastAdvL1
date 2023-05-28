import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

class DataNormalizeLayer(nn.Module):

    def __init__(self, bias, scale):

        super(DataNormalizeLayer, self).__init__()

        if isinstance(bias, float) and isinstance(scale, float):
            self._bias = torch.FloatTensor(1).fill_(bias).view(1, -1, 1, 1)
            self._scale = torch.FloatTensor(1).fill_(scale).view(1, -1, 1, 1)
        elif isinstance(bias, (tuple, list)) and isinstance(scale, (tuple, list)):
            self._bias = torch.FloatTensor(bias).view(1, -1, 1, 1)
            self._scale = torch.FloatTensor(scale).view(1, -1, 1, 1)
        else:
            raise ValueError('Invalid parameter: bias = %s, scale = %s' % (bias, scale))

    def forward(self, x):

        x = (x - self._bias.to(x.device)) / self._scale.to(x.device)

        return x
