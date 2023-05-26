import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from sonn.norm_layers import LayerNormNLP2d

def randomshift(x, shifts, learnable, max_shift, rounded_shifts, padding_mode="zeros"):
    # Take the shape of the input
    c, _, h, w = x.size()

    # Clamp the center bias in case of too much shift after back-propagation
    if learnable:
        torch.clamp(shifts, min=-max_shift, max=max_shift)

        # Round the biases to the integer values
        if rounded_shifts:
            torch.round(shifts)

    # Normalize the coordinates to [-1, 1] range which is necessary for the grid
    a_r = shifts[:,:1] / (w/2)
    b_r = shifts[:,1:] / (h/2)

    # Create the transformation matrix
    aff_mtx = torch.eye(3).to(x.device)
    aff_mtx = aff_mtx.repeat(c, 1, 1)
    aff_mtx[..., 0, 2:3] += a_r
    aff_mtx[..., 1, 2:3] += b_r

    # Create the new grid
    grid = F.affine_grid(aff_mtx[..., :2, :3], x.size(), align_corners=False)

    # Interpolate the input values
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode=padding_mode, align_corners=False)

    return x


# class LayerNormNLP(nn.Module):
#     def __init__(self, channels, affine=True):
#         super().__init__()
#         self.layer_norm = nn.LayerNorm(channels, elementwise_affine=affine)

#     def forward(self, x):
#         x = x.permute(0, 2, 3, 1)  # n c h w -> n h w c
#         x = self.layer_norm(x)
#         x = x.permute(0, 3, 1, 2)  # n h w c -> n c h w
#         return x
    
    
class SimpleGate(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    

class SimpleChannelAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, padding=0)

    def forward(self, x):
        return self.conv(self.avgpool(x))


class SuperGatedAttention2d(nn.Module):
    def __init__(self, dim, q, expand=2):
        super().__init__()
        self.dim = dim
        self.q = q
        self.expand = expand

        expanded = dim * expand
        self.conv1 = nn.Conv2d(dim, expanded, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(expanded, expanded, kernel_size=3, padding=1, groups=expanded)  # Super Neuronlar esas buraya?
        self.conv3 = nn.Conv2d(expanded // 2, dim, kernel_size=1, padding=0)

        self.conv4 = nn.Conv2d(dim, expanded, kernel_size=1, padding=0)
        self.conv5 = nn.Conv2d(expanded // 2 * q, dim, kernel_size=1, padding=0)

        self.sg = SimpleGate()
        self.sca1 = SimpleChannelAttention(expanded // 2)
        self.sca2 = SimpleChannelAttention(expanded // 2 * q)
        
        self.lnorm1 = LayerNormNLP2d(dim)
        self.lnorm2 = LayerNormNLP2d(dim)

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):

        # normalize, expand dimension, depth-wise convolve, gate + attention, reduce dimension, skip-connect.
        y = x
        y = self.lnorm1(y)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg(y)
        y = y * self.sca1(y)
        y = self.conv3(y)
        x = x + y * self.beta

        y = x
        y = self.lnorm2(y)
        y = self.conv4(y)
        y = self.sg(y)
        y = torch.cat([(y**power) for power in range(1, self.q + 1)], dim=1)
        y = y * self.sca2(y)
        y = self.conv5(y)
        x = x + y * self.gamma

        return x
    

class SuperGatedAttention2d_Plus(nn.Module):
    def __init__(self, dim, q, expand=2):
        super().__init__()
        self.dim = dim
        self.q = q
        self.expand = expand

        expanded = dim * expand
        self.conv1 = nn.Conv2d(dim, expanded, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(expanded * q, expanded, kernel_size=3, padding=1, groups=expanded)  # Super Neuronlar esas buraya?
        self.conv3 = nn.Conv2d(expanded // 2, dim, kernel_size=1, padding=0)

        self.conv4 = nn.Conv2d(dim, expanded, kernel_size=1, padding=0)
        self.conv5 = nn.Conv2d(expanded // 2 * q, dim, kernel_size=1, padding=0)

        self.sg = SimpleGate()
        self.sca1 = SimpleChannelAttention(expanded // 2)
        self.sca2 = SimpleChannelAttention(expanded // 2 * q)
        
        self.lnorm1 = LayerNormNLP2d(dim)
        self.lnorm2 = LayerNormNLP2d(dim)

        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

    def forward(self, x):

        # normalize, expand dimension, depth-wise convolve, gate + attention, reduce dimension, skip-connect.
        y = x
        y = self.lnorm1(y)
        y = self.conv1(y)
        y = torch.cat([(y**power) for power in range(1, self.q + 1)], dim=1)
        y = self.conv2(y)
        y = self.sg(y)
        y = y * self.sca1(y)
        y = self.conv3(y)
        x = x + y * self.beta

        y = x
        y = self.lnorm2(y)
        y = self.conv4(y)
        y = self.sg(y)
        y = torch.cat([(y**power) for power in range(1, self.q + 1)], dim=1)
        y = y * self.sca2(y)
        y = self.conv5(y)
        x = x + y * self.gamma

        return x










class SuperONN2d(nn.Module):
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
        learnable: bool = False,
        max_shift: float = 0,
        rounded_shifts: bool = False,
        full_mode: bool = False,
        split: int = 1,
        fill_mode: str = "zeros",
        shift_init: str = "random",
        weight_init: str = "tanh",
        verbose: bool = False
    ) -> None:
        super(SuperONN2d, self).__init__()
        self.defaults = locals().copy()
        self.defaults.pop("self", None)
        self.defaults.pop("__class__", None)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.q = q
        self.learnable = learnable
        self.max_shift = max_shift
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.rounded_shifts = rounded_shifts
        self.full_mode = full_mode
        self.split = split
        self.fill_mode = fill_mode
        self.shift_init = shift_init
        self.weight_init = weight_init
        self.verbose = verbose
        
        self.weights = nn.Parameter(torch.Tensor(self.out_channels, self.q * self.in_channels, *self.kernel_size))  # Q x C x K x D
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else: 
            self.register_parameter('bias', None)
        
        if self.learnable:
            if self.full_mode:
                self.shifts = nn.Parameter(torch.Tensor(self.in_channels * self.out_channels, 2))
            else:
                self.shifts = nn.Parameter(torch.Tensor(self.in_channels, 2))
        else:
            if self.full_mode:
                self.register_buffer('shifts', torch.Tensor(self.in_channels * self.out_channels, 2))
            else:
                self.register_buffer('shifts', torch.Tensor(self.in_channels, 2))
        
        self.reset_parameters()

        if self.verbose:
            import json
            print(f"SuperONN2d initialized with:\n{json.dumps(self.defaults, indent=4)}")

    def reset_parameters(self) -> None:
        if self.shift_init == "random":
            nn.init.uniform_(self.shifts, -self.max_shift, self.max_shift)
        elif self.shift_init == "half":
            nn.init.uniform_(self.shifts, -self.max_shift // 2, self.max_shift // 2)
        elif self.shift_init == "zeros":
            nn.init.zeros_(self.shifts)

        if self.max_shift != 0:
            with torch.no_grad():
                self.shifts.div_(self.max_shift)  ##### Normalize shifts

        if self.rounded_shifts:
            with torch.no_grad():
                self.shifts.data.round_()

        if self.weight_init == "tanh":
            gain = nn.init.calculate_gain('tanh')
            nn.init.xavier_uniform_(self.weights, gain=gain)
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
        elif self.weight_init == "selu":
            # https://github.com/bioinf-jku/SNNs
            # https://github.com/bioinf-jku/SNNs/blob/master/Pytorch/SelfNormalizingNetworks_CNN_MNIST.ipynb
            # https://github.com/bioinf-jku/SNNs/blob/master/Pytorch/SelfNormalizingNetworks_CNN_CIFAR10.ipynb
            nn.init.kaiming_normal_(self.weights, mode="fan_in", nonlinearity="linear")
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def _even_groups(self, n, g):
        g = max(min(n, g), 1)  # clamp n, 1
        rem = n % g
        numel = n // g
        groups = torch.cat((torch.tile(torch.torch.Tensor([numel + 1]), (rem,)), 
                            torch.tile(torch.torch.Tensor([numel]), (g - rem,))))
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.full_mode and self.max_shift > 0:
            ys = []
            r1 = 0
            for r2 in torch.cumsum(self._even_groups(self.out_channels, self.split), dim=0):
                r2 = int(r2.item())
                num_neurons = r2 - r1

                # stack the input out_channels number of times
                y = torch.cat([x for _ in range(num_neurons)], dim=1)  

                y = y.permute(1, 0, 2, 3)
                # each neuron applies its own shift
                y = randomshift(y, self.shifts[int(r1*self.in_channels):int(r2*self.in_channels)] * self.max_shift, self.learnable, self.max_shift, self.rounded_shifts, self.fill_mode)
                y = y.permute(1, 0, 2, 3)

                # x is N x (out_channels x in_channels) x H x W      10 x 12 x 28 x 28    ->     10 x 4 x 3 x 28 x 28    ->    10 x 4 x 6 x 28 x 28     ->   10 x 24 x 28 x 28
                # reshape to N x out_channels x in_channels x H x W  (N x in_channels x out_channels x H x W is incorrect)
                y = y.reshape(y.shape[0], num_neurons, self.in_channels, y.shape[-2], y.shape[-1])

                # add the qth dimension
                y = torch.cat([(y**i) for i in range(1, self.q + 1)], dim=2)

                # reshape to N x (out_channels x q x in_channels) x H x W
                y = y.reshape(y.shape[0], num_neurons * self.in_channels * self.q, y.shape[-2], y.shape[-1])  

                y = F.conv2d(y, self.weights[r1:r2], bias=self.bias[r1:r2] if self.bias is not None else self.bias, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=num_neurons)
                
                ys.append(y)

                r1 = r2

            return torch.cat(ys, dim=1) 
        else:
            if self.max_shift > 0:
                x = x.permute(1, 0, 2, 3)
                x = randomshift(x, self.shifts * self.max_shift, self.learnable, self.max_shift, self.rounded_shifts, self.fill_mode)  ##### Normalize back to original range!!
                x = x.permute(1, 0, 2, 3)
            
            x = torch.cat([(x**i) for i in range(1, self.q+1)],dim=1)
            x = F.conv2d(x, self.weights, bias=self.bias, padding=self.padding, stride=self.stride, dilation=self.dilation)        
            return x