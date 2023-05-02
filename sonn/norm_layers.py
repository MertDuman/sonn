import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormNLP2d(nn.Module):
    ''' 
    The name NLP implies that this layer norm does not normalize the C x H x W dimensions (default), but only the C dimension (NLP). 
    https://i.stack.imgur.com/1JdN6.png

    This is useful for NLP tasks, where the input is a sequence of word embeddings, and the C dimension is the embedding dimension.
    But it has also been used in image tasks, where the C dimension is the number of channels.
    '''
    def __init__(self, channels, affine=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(channels, elementwise_affine=affine)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # n c h w -> n h w c
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)  # n h w c -> n c h w
        return x
    

class LayerNormNLP1d(nn.Module):
    ''' 
    The name NLP implies that this layer norm does not normalize the C x H x W dimensions (default), but only the C dimension (NLP). 
    https://i.stack.imgur.com/1JdN6.png

    This is useful for NLP tasks, where the input is a sequence of word embeddings, and the C dimension is the embedding dimension.
    But it has also been used in image tasks, where the C dimension is the number of channels.
    '''
    def __init__(self, channels, affine=True):
        super().__init__()
        self.layer_norm = nn.LayerNorm(channels, elementwise_affine=affine)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # n c d -> n d c
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)  # n d c -> n c d
        return x
    

class InstanceNormalize2d(nn.Module):
    ''' Normalize each channel of each image separately with min-max normalization. '''
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
    ''' Normalize each channel of each image separately by dividing with the absolute max value. This normalizes the data range to [-1, 1] without stretching it. '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # maximum absolute value in x, so that after normalization the data range is definitely between [-1, 1] but not stretched to the range.
        maxv = torch.max(x.amax((2,3), keepdim=True).abs(), x.amin((2,3), keepdim=True).abs())
        return x / maxv