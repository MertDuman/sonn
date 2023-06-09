from collections.abc import Iterable
from itertools import repeat
import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, cat, no_grad
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out

def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)

_scalar_or_tuple_1 = Union[int, Tuple[int]]
_scalar_or_tuple_2 = Union[int, Tuple[int, int]]
_scalar_or_tuple_3 = Union[int, Tuple[int, int, int]]


class _SelfONNNd(Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'in_channels',
                     'out_channels', 'kernel_size', 'q']

    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, ...]
    q: int
    stride: Tuple[int, ...]
    padding: Tuple[int, ...]
    dilation: Tuple[int, ...]
    groups: int
    padding_mode: str
    sampling_factor: int
    dropout: float
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 split_weights: bool,
                 groups: int,
                 bias: bool,
                 q: int,
                 padding_mode,
                 mode,
                 dropout: Optional[float]):
        super(_SelfONNNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.split_weights = split_weights  # Split layer weights into q groups so that lr can be customized for each
        if padding[0] == -1:
            # Automatically calculate the needed padding for each dim
            newpadding = []
            for dimension in range(len(padding)):
                newpadding.append(math.ceil(self.kernel_size[dimension] / 2) - 1)
            self.padding = tuple(padding)
        else:
            self.padding = padding
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        self._reversed_padding_repeated_twice = tuple(x for x in reversed(self.padding) for _ in range(2))
        self.dilation = dilation
        self.groups = groups
        self.q = q
        self.padding_mode = padding_mode
        self.dropout = dropout
        valid_modes = ["fast", "low_mem"]
        if mode not in valid_modes:
            raise ValueError("mode must be one of {}".format(valid_modes))
        self.mode = mode
        if self.split_weights:
            self.weight = nn.ParameterList([nn.Parameter(Tensor(self.out_channels,self.in_channels // groups, *self.kernel_size)) for _ in range(q)]) # Q x C x K x D
        else:
            self.weight = Parameter(Tensor(
                out_channels, q * in_channels // groups, *kernel_size))
            
        if bias:
            self.bias = Parameter(Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('tanh')
        if self.split_weights:
            #nn.init.xavier_uniform_(torch.cat([weight for weight in self.weight], dim=1), gain=gain)
            for weight in self.weight: nn.init.xavier_uniform_(weight, gain=gain)
        else:
            nn.init.xavier_uniform_(self.weight, gain=gain)
        if self.bias is not None:
            if self.split_weights:
                fan_in, _ = _calculate_fan_in_and_fan_out(torch.cat([weight for weight in self.weight], dim=1))
            else:
                fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == 'fast':
            return self._forward_fast(x)
        elif self.mode == 'low_mem':
            return self._forward_low_mem(x)

    def _forward_fast(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def _forward_low_mem(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def extra_repr(self):
        repr_string = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
                       ', stride={stride}, q={q}')
        if self.padding != 0:
            repr_string += ', padding={padding}'
        if self.dilation != 1:
            repr_string += ', dilation={dilation}'
        if self.groups != 1:
            repr_string += ', groups={groups}'
        if self.bias is None:
            repr_string += ', bias=False'
        if self.padding_mode != 'zeros':
            repr_string += ', padding_mode={padding_mode}'
        return repr_string.format(**self.__dict__)

    def __setstate__(self, state):
        super(_SelfONNNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class SelfONNTranspose2d(_SelfONNNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _scalar_or_tuple_2,
                 stride: _scalar_or_tuple_2 = 1,
                 padding: _scalar_or_tuple_2 = 0,
                 output_padding: _scalar_or_tuple_2 = 0,
                 dilation: _scalar_or_tuple_2 = 1,
                 split_weights: bool = False,
                 groups: int = 1,
                 bias: bool = True,
                 q: int = 1,
                 padding_mode: str = 'zeros',
                 mode: str = 'fast',
                 dropout: Optional[float] = None) -> None:
        # Transform type from Union[int, Tuple[int, int]] to Tuple[int, int]
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = _pair(padding)
        dilation_ = _pair(dilation)
        super(SelfONNTranspose2d, self).__init__(in_channels, out_channels, kernel_size_,
                                      stride_, padding_, dilation_, split_weights, groups, bias,
                                      q, padding_mode, mode, dropout)
        self.output_padding = _pair(output_padding)     # Additional conv_transpose parameter
        
        if self.split_weights:
            self.weight = nn.ParameterList([nn.Parameter(Tensor(self.in_channels // groups, self.out_channels, *self.kernel_size)) for _ in range(q)]) # Q x C x K x D
        else:
            self.weight = Parameter(Tensor(
                q * in_channels // groups, out_channels, *self.kernel_size))

    def _forward_fast(self, x: Tensor) -> Tensor:
        x = cat([(x ** (i + 1)) for i in range(self.q)], dim=1)
        if self.dropout:
            x = F.dropout2d(x, self.dropout, self.training, False)
        if self.padding_mode != 'zeros':
            x = F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode)
            x = F.conv_transpose2d(x,
                         torch.cat([weight for weight in self.weight], 0) if self.split_weights else self.weight,
                         bias=self.bias,
                         stride=self.stride,
                         padding=_pair(0),
                         output_padding=_pair(0),
                         dilation=self.dilation,
                         groups=self.groups)
        else:
            x = F.conv_transpose2d(x,
                         torch.cat([weight for weight in self.weight], 0) if self.split_weights else self.weight,
                         bias=self.bias,
                         stride=self.stride,
                         padding=self.padding,
                         output_padding=self.output_padding,
                         dilation=self.dilation,
                         groups=self.groups)
        return x

    def _forward_low_mem(self, x: Tensor) -> Tensor:
        orig_x = x
        x = F.conv_transpose2d(orig_x,
                     self.weights[:, :self.in_channels, :, :],
                     bias=None,
                     stride=self.stride,
                     padding=self.padding,
                     output_padding=self.output_padding,
                     dilation=self.dilation)
        inchannels_per_group = self.in_channels // self.groups
        for q in range(1, self.q):
            x_to_power_q = orig_x ** (q + 1)
            if self.dropout:
                x_to_power_q = F.dropout2d(x, self.dropout, self.training, False)
            x += F.conv_transpose2d(
                x_to_power_q,
                self.weight[:, (q * inchannels_per_group):((q + 1) * inchannels_per_group), :, :],
                bias=None,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                dilation=self.dilation,
                groups=self.groups
            )
        if self.bias is not None:
            x += self.bias[None, :, None, None]
        return x


class SelfONNTranspose3d(_SelfONNNd):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _scalar_or_tuple_3,
                 stride: _scalar_or_tuple_3 = 1,
                 padding: _scalar_or_tuple_3 = 0,
                 output_padding: _scalar_or_tuple_3 = 0,
                 dilation: _scalar_or_tuple_3 = 1,
                 groups: int = 1,
                 bias: bool = True,
                 q: int = 1,
                 padding_mode: str = 'zeros',
                 mode: str = 'fast',
                 dropout: Optional[float] = None) -> None:
        # Transform type from Union[int, Tuple[int]] to Tuple[int]
        kernel_size_ = _triple(kernel_size)
        stride_ = _triple(stride)
        padding_ = _triple(padding)
        dilation_ = _triple(dilation)
        super(SelfONNTranspose3d, self).__init__(in_channels, out_channels, kernel_size_,
                                      stride_, padding_, dilation_, False, groups, bias,
                                      q, padding_mode, mode, dropout)
        self.output_padding = _triple(output_padding)     # Additional conv_transpose parameter
        self.weight = Parameter(Tensor(
            q * in_channels // groups, out_channels, *self.kernel_size))

    def forward_slow(self, x: Tensor) -> Tensor:
        # Separable w.r.t. pool operation, implementation TBD
        raise NotImplementedError

    def _forward_fast(self, x: Tensor) -> Tensor:
        x = cat([(x ** i) for i in range(1, self.q + 1)], dim=1)
        if self.padding_mode != 'zeros':
            x = F.pad(x, pad=self._reversed_padding_repeated_twice, mode=self.padding_mode)
            x = F.conv_transpose3d(x,
                         weight=self.weight,
                         bias=self.bias,
                         stride=self.stride,
                         padding=0,
                         output_padding=0,
                         dilation=self.dilation,
                         groups=self.groups)
        else:
            x = F.conv_transpose3d(x,
                         weight=self.weight,
                         bias=self.bias,
                         stride=self.stride,
                         padding=self.padding,
                         output_padding=self.output_padding,
                         dilation=self.dilation,
                         groups=self.groups)

        return x

    def _forward_low_mem(self, x: Tensor):
        orig_x = x
        x = F.conv_transpose3d(orig_x,
                     self.weights[:, :self.in_channels, :, :, :],
                     bias=None,
                     stride=self.stride,
                     padding=self.padding,
                     output_padding=self.output_padding,
                     dilation=self.dilation)
        inchannels_per_group = self.in_channels // self.groups
        for q in range(1, self.q):
            x_to_power_q = orig_x ** (q + 1)
            if self.dropout:
                x_to_power_q = F.dropout2d(x, self.dropout, self.training, False)
            x += F.conv_transpose3d(
                x_to_power_q,
                self.weight[:, (q * inchannels_per_group):((q + 1) * inchannels_per_group), :, :, :],
                bias=None,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                dilation=self.dilation,
                groups=self.groups
            )
        if self.bias is not None:
            x += self.bias[None, :, None, None]
        return x

def get_param_dict(model, base_lr, reversed=False):
    """ gets the parameter dict of a model with varying learning rates for each q component of the network (if the network has split weights)
    
    :model:     ONN model to apply varying lr to (weights must be split in to q groups)
    :base_lr:   The default learning rate, each q component will have lr=base_lr/q
    :reversed:  Higher q components have higher lr, lr=base_lr*q

    returns param_dict to be passed to pytorch optimizer
    """

    param_dict = []
    split_weights = False
    for name, param in model.named_parameters():
        # Check if there are any split lr parameters
        #print(name, param.size(), type(name), type(param))
        if '.weight.' in name:
            split_weights = True
            break

    if split_weights:
        # Set learning rates for each individual weight
        for name, param in model.named_parameters():
            if '.weight.' in name:
                # Split parameter
                q = int(name[-1]) + 1
                if reversed:
                    param_dict.append({"params": param, 'lr': base_lr*q})   # Multiply
                else:
                    param_dict.append({"params": param, 'lr': base_lr/q})   # Divide
            else:
                # Non split parameter
                param_dict.append({"params": param})
    else:
        # Set global lr in the standard way
        param_dict = [{"params": model.parameters()}]

    return param_dict


if __name__ == '__main__':
    """ Example usage with padding settings so output is exactly 2x input size"""
    
    scale = 2   # Upsamplnig scale
    ks = 3  # kernel size

    test_input = torch.rand(4,3,32,32)
    Deconv2d = SelfONNTranspose2d(3, 3, kernel_size=ks, q=3, stride=scale, padding=ks//2, output_padding=scale-1)
    out_2d = Deconv2d(test_input)
    print("2d deconv out:", out_2d.shape)

    test_input = torch.rand(4,3,32,32,32)
    Deconv3d = SelfONNTranspose3d(3,3, kernel_size=ks, q=3, stride=scale, padding=ks//2, output_padding=scale-1)
    out_3d = Deconv3d(test_input)
    print("3d deconv out:", out_3d.shape)
