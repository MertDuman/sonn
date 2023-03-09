import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceNormalize2d(nn.Module):
    def __init__(self, affine=False):
        super().__init__()
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.tensor([2.]))
            self.bias = nn.Parameter(torch.tensor([1.]))
        else:
            self.register_buffer("weight", torch.tensor([2.]))
            self.register_buffer("bias", torch.tensor([1.]))

    def forward(self, x):
        return (x - x.amin(dim=(2,3), keepdim=True)) / (x.amax(dim=(2,3), keepdim=True) - x.amin(dim=(2,3), keepdim=True)) * self.weight - self.bias


class DataRangeNormalize2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # maximum absolute value in x, so that after normalization the data range is definitely between [-1, 1] but not stretched to the range.
        maxv = torch.max(x.amax((2,3), keepdim=True).abs(), x.amin((2,3), keepdim=True).abs())
        return x / maxv