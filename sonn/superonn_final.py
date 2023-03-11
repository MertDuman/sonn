import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class SuperONN2dFactory:
    @staticmethod
    def full_mode():
        return SuperONN2d(3, 16*9, q=3, kernel_size=3, padding=1, groups=16*9, full_groups=16, max_shift=10, learnable=True)


class SuperONN2d(nn.Module):
    """
    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
    q : int
        Maclaurin series coefficient.
    bias : bool, default: True
    padding : int, default: 0
    stride : int, default: 1
    dilation : int, default: 1
    groups : int, default: 1
        Works the same as in nn.Conv2d, except (in_channels * q * full_groups) are split into groups (as opposed to in_channels).
        If groups > 1:
            in_channels * q must be divisible by (groups / full_groups), and groups must be divisible by full_groups.
        If groups == 1 and full_groups > 1:
            groups will be set to full_groups.
    shift_groups : int, default: in_channels
        The number of groups the shifts are split into. Defaults to in_channels.
        E.g. if shift_groups = 2, the first half and the second half of the input channels will have a separate set of shifts applied to them.
             Thus, by default, all input channels have a separate set of shifts applied to them.
    full_groups : int, default: 1
        Number of neuron groups in full mode. The layer will produce this many copies of the input, where each one is shifted independently.
        When 1, the layer works in semi-mode, where the shifts are shared across neurons.
        When out_channels, the layer works in full-mode, where each neuron has a separate set of shifts applied to the input.
        Anything in-between is a performance vs. runtime & memory trade-off.
        Not to be confused with shift_groups, which determines the shift independence of channels.
        Dramatically slows down training, but may provide better learning capabilities.
    learnable : bool, default: True
        If True, shifts are optimized with backpropagation.
    max_shift : float, default: 0
        Maximum possible value the shifts can take.
    rounded_shifts : bool, default: False
        Whether shifts are rounded to the nearest integer.
    split_iterations : int, default: 1
        When training in full_mode (e.g. full_groups > 1), memory can be an issue, as the input needs to be copied many times.
        When split_iterations > 1, the forward pass is done in 'split_iterations' amount of iterations, increasing runtime but lowering memory consumption.
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
    dtype : str, default: None
        The datatype of the weights. If None, the default datatype is used.
    verbose : bool, default: False
        Whether to print the parameters of the layer.
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
        groups: int = 1,
        shift_groups: int = None,
        full_groups: int = 1,
        learnable: bool = False,
        max_shift: float = 0,
        rounded_shifts: bool = False,
        split_iterations: int = 1,
        fill_mode: str = "zeros",
        shift_init: str = "random",
        weight_init: str = "tanh",
        dtype: str = None,
        verbose: bool = False
    ) -> None:
        super().__init__()

        # Ensures that a neuron does not process channels belonging to different full groups.
        groups = full_groups if groups == 1 else groups
        assert groups % full_groups == 0, f"groups ({groups}) must be divisible by full_groups ({full_groups})"
        assert shift_groups is None or in_channels % shift_groups == 0, f"in_channels ({in_channels}) must be divisible by shift_groups ({shift_groups})"
        assert (in_channels * q) % (groups / full_groups) == 0, f"in_channels * q ({in_channels * q}) must be divisible by groups / full_groups ({groups // full_groups})"
        assert out_channels % groups == 0, f"out_channels ({out_channels}) must be divisible by groups ({groups})"

        # Ensures that a neuron does not process channels raised to different q-powers. This may be enforced in the future.
        # assert (groups // full_groups) % q == 0, f"groups / full_groups ({groups // full_groups}) must be divisible by q ({q})"

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
        self.groups = groups
        self.shift_groups = in_channels if shift_groups is None else shift_groups
        self.full_groups = full_groups
        self.learnable = learnable
        self.max_shift = max_shift
        self.rounded_shifts = rounded_shifts
        self.split_iterations = split_iterations
        self.fill_mode = fill_mode
        self.shift_init = shift_init
        self.weight_init = weight_init
        self.dtype = dtype
        self.verbose = verbose
        
        neuron_depth = (in_channels * q) // (groups // full_groups)
        self.weight = nn.Parameter(torch.empty(self.out_channels, neuron_depth, *self.kernel_size, dtype=dtype))  # Q x C x K x D
        self.bias = nn.Parameter(torch.empty(self.out_channels)) if bias else self.register_parameter('bias', None)
            
        if self.learnable:
            self.shifts = nn.Parameter(torch.empty(self.full_groups, self.shift_groups, 2))
        else:
            self.register_buffer('shifts', torch.empty(self.full_groups, self.shift_groups, 2))
        
        self.reset_parameters()

        if self.verbose:
            import json
            print(f"SuperONN2d initialized with:\n{json.dumps(self.defaults, indent=4)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, _, H, W = x.shape

        if self.full_groups > 1:
            # stack the input full_groups number of times
            x = torch.cat([x for _ in range(self.full_groups)], dim=1)

        # No need to shift the input if max_shift is 0
        if self.max_shift > 0:
            # (full_groups, shift_groups, 2) => (full_groups, in_channels, 2) => (full_groups x in_channels, 2)
            shifts = torch.repeat_interleave(self.shifts, self.in_channels // self.shift_groups, dim=1).reshape(self.full_groups * self.in_channels, 2)
            x = x.permute(1, 0, 2, 3)
            x = randomshift(x, shifts * self.max_shift, self.learnable, self.max_shift, self.rounded_shifts, self.fill_mode)  ##### Normalize back to original range!!
            x = x.permute(1, 0, 2, 3)

        if self.full_groups > 1:
            # (N, full_groups * in_channels, H, W) => (N, full_groups, in_channels, H, W) => 
            # (N, full_groups, q * in_channels, H, W) => (N, full_groups * q * in_channels, H, W)
            # We have full_groups many q many in_channels x H x W tensors.
            x = x.reshape(N, self.full_groups, self.in_channels, H, W)
            x = torch.cat([(x**i) for i in range(1, self.q + 1)], dim=2)
            x = x.reshape(N, self.full_groups * self.in_channels * self.q, H, W) 
        else:
            x = torch.cat([(x**i) for i in range(1, self.q + 1)], dim=1) 
        
        #x = torch.cat([(x**i) for i in range(1, self.q+1)],dim=1)
        x = F.conv2d(x, self.weight, bias=self.bias, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)        
        return x
        
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

    def _even_groups(self, n, g):
        """Evenly split n into g groups such that the first n % g groups have n // g + 1 elements and the rest have n // g elements."""
        g = max(min(n, g), 1)  # clamp n, 1
        rem = n % g
        numel = n // g
        groups = torch.cat((torch.tile(torch.torch.Tensor([numel + 1]), (rem,)), 
                            torch.tile(torch.torch.Tensor([numel]), (g - rem,))))
        return groups
    
    def extra_repr(self) -> str:
        repr_string = '{in_channels}, {out_channels}, q={q}, kernel_size={kernel_size}, padding={padding}, stride={stride}, dilation={dilation}, max_shift={max_shift}, learnable={learnable}, groups={groups}, full_groups={full_groups}, shift_groups={shift_groups}'
        return repr_string.format(**self.__dict__)

