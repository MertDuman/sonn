
import math
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.ops



class SuperONN2d_v2(nn.Module):
    """
    Parameters
    ----------
    in_channels
    out_channels
    kernel_size
    q
        Maclaurin series coefficient.
    bias
    padding
    stride
    dilation
    learnable
        If True, shift terms are updated through backpropagation.
    max_shift
        Maximum possible value the shifts can take.
    rounded_shifts
        Whether shifts are rounded to the nearest integer.
    full_mode
        Whether each neuron has a separate set of shifts. Dramatically slows down training, but may provide better learning capabilities.
    split
        When training in full_mode, memory can be an issue, as every neuron has a separate set of shifts applied to the input, effectively multiplying the input size.
        When split > 1, the forward pass is done in split amount of iterations, increasing runtime but lowering memory consumption.
    fill_mode : one of ['zeros', 'reflection', 'border'], default: 'zeros'
        The fill applied when the input is shifted, thus removing some pixels.
        When 'zeros', the removed pixels are filled with zero.
        When 'reflection', the removed pixels are filled with values when the input is reflected at the border.
        When 'border', the removed pixels are filled with values at the border.
    shift_init : one of ['random', 'half', 'zeros'], default: 'random'
        The default value the shifts take. When learnable is False and shift_init is 'zeros', this is identical to SelfONNs.
        When 'zeros', shifts start at zero.
        When 'random', shifts are uniformly distributed between [-max_shift, max_shift].
        When 'half', shifts are uniformly distributed between [-max_shift / 2, max_shift / 2].
    weight_init : one of ['tanh', 'selu'], default: 'tanh'
        Determines how to initialize the weights based on the activation function used in the network. If unsure, keep as default.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        q: int, 
        bias: bool = True,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        shift_groups: int = 1,
        learnable: bool = False,
        max_shift: float = 0,
        rounded_shifts: bool = False,
        fill_mode: str = "zeros",
        weight_init: str = "tanh",
        verbose: bool = False
    ) -> None:
        super().__init__()
        self.defaults = locals().copy()
        self.defaults.pop("self", None)
        self.defaults.pop("__class__", None)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.q = q
        
        self.padding = padding
        self.stride = stride
        self.dilation = dilation

        self.shift_groups = shift_groups
        self.learnable = learnable
        self.max_shift = max_shift
        self.rounded_shifts = rounded_shifts
        self.fill_mode = fill_mode
        self.weight_init = weight_init
        self.verbose = verbose
        
        self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.q * self.in_channels, *self.kernel_size))  # Q x C x K x D
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else: 
            self.register_parameter('bias', None)
        
        if self.learnable:
            # Calculating shifts through convolution
            self.shift_conv = nn.Conv2d(
                in_channels, 
                2 * self.shift_groups,
                kernel_size=self.kernel_size, 
                stride=self.stride,
                padding=self.padding, 
                bias=True
            )

            nn.init.constant_(self.shift_conv.weight, 0.)
            nn.init.constant_(self.shift_conv.bias, 0.)
        else:
            # Deprecated for now
            self.register_buffer('shifts', torch.Tensor(self.in_channels * self.out_channels, 2))
        
        self.reset_parameters()

        if self.verbose:
            import json
            print(f"SuperONN2d initialized with:\n{json.dumps(self.defaults, indent=4)}")

    def reset_parameters(self) -> None:
        if self.weight_init == "tanh":
            gain = nn.init.calculate_gain('tanh')
            nn.init.xavier_uniform_(self.weight, gain=gain)
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
        elif self.weight_init == "selu":
            # https://github.com/bioinf-jku/SNNs
            # https://github.com/bioinf-jku/SNNs/blob/master/Pytorch/SelfNormalizingNetworks_CNN_MNIST.ipynb
            # https://github.com/bioinf-jku/SNNs/blob/master/Pytorch/SelfNormalizingNetworks_CNN_CIFAR10.ipynb
            nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="linear")
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # N x (2 x Groups) x OutH x OutW
        shifts = self.shift_conv(x).clamp(-self.max_shift, self.max_shift)

        # Needs to be N x (2 x Groups x KernelH x KernelW) x OutH x OutW
        shifts = torch.repeat_interleave(shifts, self.kernel_size[0] * self.kernel_size[1], 1)
        
        x = torch.cat([(x**i) for i in range(1, self.q+1)], dim=1)
        x = torchvision.ops.deform_conv2d(
            input=x, 
            offset=shifts, 
            weight=self.weight, 
            bias=self.bias, 
            padding=self.padding,
            stride=self.stride,
        )
        return x





class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeformableConv2d, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        
        self.modulator_conv = nn.Conv2d(in_channels, 
                                     1 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        
        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        #h, w = x.shape[2:]
        #max_offset = max(h, w)/4.

        offset = self.offset_conv(x)#.clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        
        x = torchvision.ops.deform_conv2d(input=x, 
                                          offset=offset, 
                                          weight=self.regular_conv.weight, 
                                          bias=self.regular_conv.bias, 
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return 