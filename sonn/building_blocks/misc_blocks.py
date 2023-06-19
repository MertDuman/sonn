import torch
import torch.nn as nn
import torch.nn.functional as F
from sonn.superonn_final import SuperONN2d
from sonn.norm_layers import LayerNormNLP2d


class Downsample2d(nn.Module):
    ''' Downsample H and W by 2, upsample C by 2. '''
    def __init__(self, dim, kernel_size=3, shuffle_first=False):
        super().__init__()
        if shuffle_first:
            self.layers = nn.Sequential(
                nn.PixelUnshuffle(2),
                nn.Conv2d(dim * 4, dim * 2, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(dim, dim // 2, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
                nn.PixelUnshuffle(2)
            )
    def forward(self, x):
        return self.layers(x)
    

class DownsampleONN2d(nn.Module):
    ''' Downsample H and W by 2, upsample C by 2. '''
    def __init__(self, dim, kernel_size=3, shuffle_first=False, **onn_kwargs):
        super().__init__()
        if shuffle_first:
            self.layers = nn.Sequential(
                nn.PixelUnshuffle(2),
                LayerNormNLP2d(dim * 4),
                SuperONN2d(dim * 4, dim * 2, kernel_size=kernel_size, padding=kernel_size // 2, bias=False, **onn_kwargs),
            )
        else:
            self.layers = nn.Sequential(
                LayerNormNLP2d(dim),
                SuperONN2d(dim, dim // 2, kernel_size=kernel_size, padding=kernel_size // 2, bias=False, **onn_kwargs),
                nn.PixelUnshuffle(2)
            )
    def forward(self, x):
        return self.layers(x)


class Upsample2d(nn.Module):
    ''' Upsample H and W by 2, downsample C by 2. '''
    def __init__(self, dim, kernel_size=3, shuffle_first=True):
        super().__init__()
        if shuffle_first:
            self.layers = nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(dim // 4, dim // 2, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(dim, dim * 2, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
                nn.PixelShuffle(2)
            )
    def forward(self, x):
        return self.layers(x)
    

class UpsampleONN2d(nn.Module):
    ''' Upsample H and W by 2, downsample C by 2. '''
    def __init__(self, dim, kernel_size=3, shuffle_first=True, **onn_kwargs):
        super().__init__()
        if shuffle_first:
            self.layers = nn.Sequential(
                nn.PixelShuffle(2),
                LayerNormNLP2d(dim // 4),
                SuperONN2d(dim // 4, dim // 2, kernel_size=kernel_size, padding=kernel_size // 2, bias=False, **onn_kwargs),
            )
        else:
            self.layers = nn.Sequential(
                LayerNormNLP2d(dim),
                SuperONN2d(dim, dim * 2, kernel_size=kernel_size, padding=kernel_size // 2, bias=False, **onn_kwargs),
                nn.PixelShuffle(2)
            )
    def forward(self, x):
        return self.layers(x)
    

class SimpleGate(nn.Module):
    ''' Halves the number of channels by multiplying the first half with the second half. '''
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2