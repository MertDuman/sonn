import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from collections.abc import Iterable 
from typing import Union


def randomshift_1d(x, shifts, learnable, max_shift, rounded_shifts, padding_mode="zeros"):
    """ Since grid_sample doesn't work in 1d, we just add a dummy width of 1 to our signals. """
    x = x.moveaxis(0, 1)
    x = x.unsqueeze(-1)
    # Take the shape of the input
    c, _, h, w = x.size()

    # Clamp the center bias in case of too much shift after back-propagation
    if learnable:
        torch.clamp(shifts, min=-max_shift, max=max_shift)

        # Round the biases to the integer values
        if rounded_shifts:
            torch.round(shifts)

    # Normalize the coordinates to [-1, 1] range which is necessary for the grid
    a_r = torch.zeros_like(shifts)
    b_r = shifts / (h / 2)

    # Create the transformation matrix
    aff_mtx = torch.eye(3).to(x.device)
    aff_mtx = aff_mtx.repeat(c, 1, 1)
    aff_mtx[..., 0, 2:3] += a_r
    aff_mtx[..., 1, 2:3] += b_r

    # Create the new grid
    grid = F.affine_grid(aff_mtx[..., :2, :3], x.size(), align_corners=False)

    # Interpolate the input values
    x = F.grid_sample(x, grid, mode='bilinear', padding_mode=padding_mode, align_corners=False)
    x = x.squeeze(-1)
    x = x.moveaxis(0, 1)
    return x


def take_qth_power(x, q, with_w0=False):
    if x.ndim == 3:
        N, C, F = x.shape
    elif x.ndim == 4:
        N, FG, C, F = x.shape
    start = 0 if with_w0 else 1
    total = q+1 if with_w0 else q
    x = torch.cat([x**i for i in range(start, q+1)], dim=1)
    if x.ndim == 3:
        x = x.reshape(N, total, C, F).transpose(1, 2).reshape(N, C*total, F)
    elif x.ndim == 4:
        x = x.reshape(N, FG, total, C, F).transpose(2, 3).reshape(N, FG, C*total, F)
    return x


class SuperONN1d(nn.Module):
    """
    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
    q : int
        Order of the Maclaurin series: w0 + w1 * x + w2 * x^2 + ... + wq * x^q
    bias : bool
    with_w0 : bool
        Whether to include a separate bias term for the Maclaurin series.
        Remember the Maclaurin series: w0 + w1 * x + w2 * x^2 + ... + wq * x^q
        When with_w0 is False, w0 is the bias term that all neurons share.
        When with_w0 is True, w0 is a separate bias term for each neuron. The computational effect of settings this to True is similar to increasing q by 1.
    padding : int
    stride : int
    dilation : int
    groups : int or iterable of int or str
        Works the same as in nn.Conv2d, except (in_channels * q * full_groups) are split into groups (as opposed to in_channels).
        If groups > 1:
            in_channels * q must be divisible by (groups / full_groups), and groups must be divisible by full_groups.
        If groups == 1 and full_groups > 1:
            groups will be set to full_groups.
        If groups == 'depthwise', each channel will be processed by one neuron (groups == in_channels). Note that if q > 1, this corresponds to each maclaurin series being processed by one neuron.
        NOTE:
            Additionally, groups can be an iterable of int. If so, it must be of length out_channels, and the sum of its elements must be in_channels * q * full_groups.
            This allows for more fine-grained control over the groups, and allows each neuron to process a different amount of channels.
            !! Slows down training on the order of unique ints in groups, but may provide better learning capabilities.
    shift_groups : int, default: in_channels
        The number of groups the shifts are split into. Defaults to in_channels.
        E.g. if shift_groups = 2, the first half and the second half of the input channels will have a separate set of shifts applied to them.
             Thus, by default, all input channels have a separate set of shifts applied to them.
    full_groups : int
        Number of neuron groups in full mode. The layer will produce this many copies of the input, where each one is shifted independently.
        When 1, the layer works in semi-mode, where the shifts are shared across neurons.
        When out_channels, the layer works in full-mode, where each neuron has a separate set of shifts applied to the input.
        Anything in-between is a performance vs. runtime & memory trade-off.
        Not to be confused with shift_groups, which determines the shift independence of channels.
        !! Dramatically slows down training, but may provide better learning capabilities.
    learnable : bool
        If True, shifts are optimized with backpropagation.
    max_shift : float or str
        Maximum possible value the shifts can take. You can pass 'auto' to set max_shift to sqrt(out_channels) // 2.
        Some common values:     Channels            Shift
                                [4, 16)             1
                                [16, 36)            2
                                [36, 64)            3
                                [64, 100)           4
                                [100, 144)          5
                                [256, 324)          8
                                [1024, 1156)        16
    rounded_shifts : bool
        Whether shifts are rounded to the nearest integer.
    split_iterations : int
        When training in full_mode (e.g. full_groups > 1), memory can be an issue, as the input needs to be copied many times.
        When split_iterations > 1, the forward pass is done in 'split_iterations' amount of iterations, increasing runtime but lowering memory consumption.
    fill_mode : one of ['zeros', 'reflection', 'border']
        The fill applied when the input is shifted, thus removing some pixels.
        When 'zeros', the removed pixels are filled with zero.
        When 'reflection', the removed pixels are filled with values when the input is reflected at the border.
        When 'border', the removed pixels are filled with values at the border.
    shift_init : one of ['random', 'half', 'zeros']
        The default value the shifts take. When learnable is False and shift_init is 'zeros', this is identical to SelfONNs.
        When 'zeros', shifts start at zero.
        When 'random', shifts are uniformly distributed between [-max_shift, max_shift].
        When 'random_int', shifts are uniformly distributed between [-max_shift, max_shift] rounded to closest integer.
        When 'half', shifts are uniformly distributed between [-max_shift / 2, max_shift / 2].
        When 'half_int', shifts are uniformly distributed between [-max_shift / 2, max_shift / 2] rounded to closest integer.
    weight_init : one of ['tanh', 'selu']
        Determines how to initialize the weights based on the activation function used in the network. If unsure, keep as default.
    dtype : str
        The datatype of the weights. If None, the default datatype is used.
    verbose : bool
        Whether to print the parameters of the layer.
    new_version : bool
        Whether to use the new version of stacking Q powers of x. This is required if using group convolutions, so that a neuron processes the powers of the same channel.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        q: int, 
        bias: bool = True,
        with_w0: bool = False,
        padding: int = 0,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        shift_groups: int = None,
        full_groups: int = 1,
        learnable: bool = False,
        max_shift: Union[float, str] = 0,
        rounded_shifts: bool = False,
        split_iterations: int = 1,
        fill_mode: str = "zeros",
        shift_init: str = "random_int",
        weight_init: str = "tanh",
        dtype: str = None,
        verbose: bool = False,
        new_version: bool = False,
    ) -> None:
        super().__init__()
        # Handle defaults
        shift_groups = in_channels if shift_groups is None else shift_groups
        
        if groups == 1:
            # Ensures that a neuron does not process channels belonging to different full groups.
            groups = full_groups
        elif groups == "depthwise":
            groups = in_channels * full_groups
            
        if max_shift == "auto":
            max_shift = math.sqrt(out_channels) // 2

        self._impl_q = q + 1 if with_w0 else q
        impl_q_str = "(q + 1)" if with_w0 else "q"
        if isinstance(groups, Iterable):
            assert all(isinstance(g, int) for g in groups), f"groups must be an iterable of int, but got {groups}"
            assert len(groups) == out_channels, f"groups must be of length out_channels ({out_channels}), but got {len(groups)}"
            assert sum(groups) == in_channels * self._impl_q * full_groups, f"sum of groups ({sum(groups)}) must be in_channels * {impl_q_str} * full_groups ({in_channels * self._impl_q * full_groups})"
        else:
            assert out_channels % groups == 0, f"out_channels ({out_channels}) must be a multiple of groups ({groups})"

            # Ensures that a neuron does not process channels belonging to different full groups.
            assert groups % full_groups == 0, f"groups ({groups}) must be a multiple of full_groups ({full_groups}), so that neurons do not process channels belonging to different full groups"
            assert (in_channels * self._impl_q * full_groups) % groups == 0, f"in_channels * {impl_q_str} * full_groups ({in_channels * self._impl_q * full_groups}) must be a multiple of groups ({groups})"
            
            # Ensures that a neuron does not process channels belonging to a different maclaurin series.
            num_el = (in_channels * self._impl_q * full_groups) // groups
            if num_el % self._impl_q != 0:
                pass
                #warnings.warn(f"Channels per group ({num_el}) must be a multiple of {impl_q_str} ({self.impl_q}), so that neurons do not process channels belonging to a different maclaurin series", source=SuperONN2d)

        assert shift_groups is None or in_channels % shift_groups == 0, f"in_channels ({in_channels}) must be divisible by shift_groups ({shift_groups})"

        self.defaults = locals().copy()
        self.defaults.pop("self", None)
        self.defaults.pop("__class__", None)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size,)
        self.q = q
        self.use_bias = bias
        self.with_w0 = with_w0
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self._iterable_groups = isinstance(groups, Iterable)
        self.shift_groups = shift_groups
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
        self.new_version = new_version
        
        if isinstance(self.groups, Iterable):
            # This branch is almost never taken, as groups are rarely an iterable.
            self._create_weights_for_iterable_groups()
        else:
            neuron_depth = (in_channels * self._impl_q * full_groups) // groups  # (in_channels * q * full_groups) / groups
            self.weight = nn.Parameter(torch.empty(self.out_channels, neuron_depth, *self.kernel_size, dtype=dtype))  # Q x C x K x D
            self.bias = nn.Parameter(torch.empty(self.out_channels)) if self.use_bias else self.register_parameter('bias', None)
            
        if self.learnable:
            self.shifts = nn.Parameter(torch.empty(self.full_groups, self.shift_groups, 1))
        else:
            # TODO: if max_shift is 0, there is no need to store the shifts
            self.register_buffer('shifts', torch.empty(self.full_groups, self.shift_groups, 1))
        
        self.reset_parameters()

        # TODO: Slight memory overhead to experiment with shifts. Allows us to see how much the shifts change during training.
        with torch.no_grad():
            self.register_buffer('initial_shifts', self.shifts.clone())

        if self.verbose:
            import json
            print(f"SuperONN2d initialized with:\n{json.dumps(self.defaults, indent=4)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Subclass SuperONN2d where each subclass handles one of the if statements below. Might be faster if we don't branch in the forward pass.
        N, _, L = x.shape

        if self.full_groups > 1:
            # stack the input full_groups number of times
            x = torch.cat([x for _ in range(self.full_groups)], dim=1)

        # No need to shift the input if max_shift is 0
        if self.max_shift > 0:
            # (full_groups, shift_groups, 2) => (full_groups, in_channels, 2) => (full_groups x in_channels, 2)
            shifts = torch.repeat_interleave(self.shifts, self.in_channels // self.shift_groups, dim=1).reshape(self.full_groups * self.in_channels, 1)
            x = randomshift_1d(x, shifts * self.max_shift, self.learnable, self.max_shift, self.rounded_shifts, self.fill_mode)  ##### Normalize back to original range!!

        if self.full_groups > 1:
            # (N, full_groups * in_channels, L) => (N, full_groups, in_channels, L) =>
            # (N, full_groups, q * in_channels, L) => (N, full_groups * q * in_channels, L)
            # We have full_groups many q many in_channels x L tensors.
            x = x.reshape(N, self.full_groups, self.in_channels, L)
            if self.new_version:
                x = take_qth_power(x, self.q, with_w0=self.with_w0)
            else:
                x = torch.cat([(x**i) for i in range(0 if self.with_w0 else 1, self.q + 1)], dim=2)
            x = x.reshape(N, self.full_groups * self.in_channels * (self.q if not self.with_w0 else (self.q + 1)), L)
        else:
            if self.new_version:
                x = take_qth_power(x, self.q, with_w0=self.with_w0)
            else:
                x = torch.cat([(x**i) for i in range(0 if self.with_w0 else 1, self.q + 1)], dim=1) 
        
        if self._iterable_groups:
            x = self._process_iterable_groups(x)
        else:
            x = F.conv1d(x, self.weight, bias=self.bias, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)
        return x
        
    def reset_parameters(self) -> None:
        if self.shift_init == "random":
            nn.init.uniform_(self.shifts, -self.max_shift, self.max_shift)
        elif self.shift_init == "random_int":
            nn.init.uniform_(self.shifts, -self.max_shift, self.max_shift)
            with torch.no_grad():
                torch.round_(self.shifts)
        elif self.shift_init == "half":
            nn.init.uniform_(self.shifts, -self.max_shift // 2, self.max_shift // 2)
        elif self.shift_init == "half_int":
            nn.init.uniform_(self.shifts, -self.max_shift // 2, self.max_shift // 2)
            with torch.no_grad():
                torch.round_(self.shifts)
        elif self.shift_init == "zeros":
            nn.init.zeros_(self.shifts)

        if self.max_shift != 0:
            with torch.no_grad():
                self.shifts.div_(self.max_shift)  ##### Normalize shifts

        if self.rounded_shifts:
            with torch.no_grad():
                self.shifts.data.round_()

        # TODO: Different weight initialization for different groups, maybe concat the weights and biases and then initialize in one go?
        if isinstance(self.groups, Iterable):
            for i in range(self._num_unique_groups):
                self.reset_weight(self.weight[i], self.bias)
        else:
            self.reset_weight(self.weight, self.bias)

    def reset_weight(self, weight, bias):
        if self.weight_init == "tanh":
            gain = nn.init.calculate_gain('tanh')
            nn.init.xavier_uniform_(weight, gain=gain)
            if bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(bias, -bound, bound)
        elif self.weight_init == "selu":
            # https://github.com/bioinf-jku/SNNs
            # https://github.com/bioinf-jku/SNNs/blob/master/Pytorch/SelfNormalizingNetworks_CNN_MNIST.ipynb
            # https://github.com/bioinf-jku/SNNs/blob/master/Pytorch/SelfNormalizingNetworks_CNN_CIFAR10.ipynb
            nn.init.kaiming_normal_(weight, mode="fan_in", nonlinearity="linear")
            if bias is not None:
                nn.init.zeros_(bias)

    def _even_groups(self, n, g):
        """ Evenly split n into g groups such that the first n % g groups have n // g + 1 elements and the rest have n // g elements. """
        g = max(min(n, g), 1)  # clamp n, 1
        rem = n % g
        numel = n // g
        groups = torch.cat((torch.tile(torch.torch.Tensor([numel + 1]), (rem,)), 
                            torch.tile(torch.torch.Tensor([numel]), (g - rem,))))
        return groups
    
    def _create_weights_for_iterable_groups(self):
        """ 
        Creates weights and biases in such a way that an iterable number of groups are processed efficiently. This is accomplished by 
        processing neurons with the same group size in parallel. This is still highly inefficient, effectively creating N number of layers instead of 1,
        where N is the number of unique values in groups.
        """
        # Find the unique values in groups, e.g. [1,3,2,2,3] -> [1,2,3] and get their inverse indices, e.g. [0,2,1,1,2].
        # Inverse indices tells us which neurons are processed together., e.g. neuron 0 is in set 0, neurons 2,3 are in set 1, and neurons 1,4 are in set 2.
        # Inverse indices are also used to correctly resort the output channels back to their original order in forward, as we don't process them in order.
        val_unique, self._out_idx_map = torch.unique(torch.tensor(self.groups), return_inverse=True)
        # This duplicates the inverse indices, e.g. [0,2,1,1,2] -> [0,2,2,2,1,1,1,1,2,2,2] allowing us to use it as an index map for the input channels.
        # E.g. set 0 processes channel 0, set 1 processes channels 4-7, set 2 processes channels 1-3 and 8-10.
        self._in_idx_map = torch.repeat_interleave(self._out_idx_map, torch.tensor(self.groups))
        self._num_unique_groups = len(val_unique)
        # We have a different set of weights for each unique group. Neurons that process the same number of channels are in the same set.
        # Now we can process neurons with the same group size in parallel. We are looping 3 times instead of 5 times for the example above.
        # Considering out_channels are usually >=64, this is a significant speedup.
        self.weight = nn.ParameterList()
        for i in range(self._num_unique_groups):
            out_idx = torch.where(self._out_idx_map == i)[0]  # indices of the neurons in set i <--> indices of the output channels
            neuron_depth = val_unique[i]  # neurons in set i process val_unique[i] many channels each
            self.weight.append(nn.Parameter(torch.empty(len(out_idx), neuron_depth, *self.kernel_size, dtype=self.dtype)))
            self.bias = nn.Parameter(torch.empty(self.out_channels)) if self.use_bias else self.register_parameter('bias', None)

    def _process_iterable_groups(self, x):
        """ Handles iterable groups in a way that neurons with the same group size are processed in parallel. """
        y = []
        resort_idx = []
        for i in range(self._num_unique_groups):
            in_idx = torch.where(self._in_idx_map == i)[0]  # neurons in set i process channels in_idx
            out_idx = torch.where(self._out_idx_map == i)[0]  # indices of the neurons in set i <--> indices of the output channels
            inp = x[:, in_idx, :]
            out = F.conv1d(inp, self.weight[i], bias=self.bias[out_idx], padding=self.padding, stride=self.stride, dilation=self.dilation, groups=len(out_idx))
            y.append(out)
            resort_idx.extend(out_idx.tolist())  # keep track of the order of the output channels, as we don't process them in order
        y = torch.cat(y, dim=1)
        resort_idx = torch.argsort(torch.tensor(resort_idx))
        return y[:, resort_idx, :]  # resort the output channels to their original order
    
    def extra_repr(self) -> str:
        repr_string = '{in_channels}, {out_channels}, q={q}, kernel_size={kernel_size}, padding={padding}, stride={stride}, dilation={dilation}, max_shift={max_shift}, learnable={learnable}, groups={groups}, full_groups={full_groups}, shift_groups={shift_groups}'
        return repr_string.format(**self.__dict__)

