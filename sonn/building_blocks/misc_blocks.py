import torch
import torch.nn as nn
import torch.nn.functional as F
from sonn.superonn_final import SuperONN2d


class Downsample2d(nn.Module):
    ''' Downsample H and W by 2, upsample C by 2. '''
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )
    def forward(self, x):
        return self.layers(x)
    

class Downsample2d_ONN(nn.Module):
    ''' Downsample H and W by 2, upsample C by 2. '''
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            SuperONN2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, bias=False, **kwargs),
            nn.PixelUnshuffle(2)
        )
    def forward(self, x):
        return self.layers(x)
    

class Downsample2d_ReverseOrder(nn.Module):
    ''' Downsample H and W by 2, upsample C by 2. Rarely used. First applies PixelUnshuffle, then Conv2d. '''
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(dim * 4, dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
        )
    def forward(self, x):
        return self.layers(x)


class Upsample2d(nn.Module):
    ''' Upsample H and W by 2, downsample C by 2. '''
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )
    def forward(self, x):
        return self.layers(x)


class Upsample2d_ONN(nn.Module):
    ''' Upsample H and W by 2, downsample C by 2. '''
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.layers = nn.Sequential(
            SuperONN2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, bias=False, **kwargs),
            nn.PixelShuffle(2)
        )
    def forward(self, x):
        return self.layers(x)
    

class SimpleGate(nn.Module):
    ''' Halves the number of channels by multiplying the first half with the second half. '''
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2