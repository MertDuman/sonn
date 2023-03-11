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


class SuperONN1d(nn.Module):
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
        padding_mode: str = "zeros"
    ) -> None:
        super(SuperONN1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size,)
        self.q = q
        self.learnable = learnable
        self.max_shift = max_shift
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.rounded_shifts = rounded_shifts
        self.full_mode = full_mode
        self.split = split
        self.padding_mode = padding_mode
        
        self.weights = nn.Parameter(torch.Tensor(self.out_channels, self.q*self.in_channels, *self.kernel_size)) # Q x C x K x D
        
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
                self.register_buffer('shifts',torch.Tensor(self.in_channels * self.out_channels, 2))
            else:
                self.register_buffer('shifts',torch.Tensor(self.in_channels, 2))
        
        self.reset_parameters()
        print("SuperONNLayer1D initialized with shifts:",max_shift, self.rounded_shifts, self.q)

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.shifts, -self.max_shift // 2, self.max_shift // 2)
        with torch.no_grad():
            self.shifts.div_(self.max_shift)  ##### Normalize shifts

        with torch.no_grad():
            if self.rounded_shifts: self.shifts.data.round_()
                
            # Zero out the y-component of shift TODO: This should be done in a better way!
            self.shifts[:,1:].data.zero_()

        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.weights, gain=gain)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NEEDS FULL MODE IMPLEMENTATION
        x = x.permute(1, 0, 2).unsqueeze(-2)
        x = randomshift(x, self.shifts * self.max_shift, self.learnable, self.max_shift, self.rounded_shifts)
        x = x.permute(1, 0, 2, 3).squeeze(-2)
        
        if self.q != 1: x = torch.cat([(x**i) for i in range(1, self.q+1)], dim=1)
        x = F.conv1d(x, self.weights, bias=self.bias, padding=self.padding, stride=self.stride, dilation=self.dilation)        
        return x

    def extra_repr(self) -> str:
        repr_string = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
                       ', q={q}, max_shift={max_shift}')
        return repr_string.format(**self.__dict__)


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

    def extra_repr(self) -> str:
        repr_string = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
                       ', q={q}, max_shift={max_shift}')
        return repr_string.format(**self.__dict__)


# Deprecated for now.
class SuperONNTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        q, 
        bias=True,
        stride=1,
        padding=0,
        output_padding=0,
        dilation=1,
        learnable=False,
        max_shift=0,
        rounded_shifts=False,
        full_mode=False,
        split=1,
        padding_mode: str = "zeros"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.q = q
        self.learnable = learnable
        self.max_shift = max_shift
        self.padding = padding
        self.stride = stride
        self.output_padding = output_padding
        self.dilation = dilation
        self.rounded_shifts = rounded_shifts
        self.full_mode = full_mode
        self.split = split
        self.padding_mode = padding_mode
        
        self.weights = nn.Parameter(torch.Tensor(self.out_channels, self.q * self.in_channels, *self.kernel_size)) # Q x C x K x D
        
        if bias: self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else: self.register_parameter('bias', None)
        
        if self.learnable:
            if self.full_mode:
                self.shifts = nn.Parameter(torch.Tensor(self.in_channels * self.out_channels, 2))
            else:
                self.shifts = nn.Parameter(torch.Tensor(self.in_channels, 2))
        else:
            if self.full_mode:
                self.register_buffer('shifts',torch.Tensor(self.in_channels * self.out_channels, 2))
            else:
                self.register_buffer('shifts',torch.Tensor(self.in_channels, 2))
        
        self.reset_parameters()
        print(f"SuperONNTranspose2d initialized with shifts={max_shift}, rounded={rounded_shifts}, q={q}, learnable={learnable}, full_mode={full_mode}")
    
    def forward(self, x):
        if self.full_mode:
            ys = []
            r1 = 0
            for r2 in torch.cumsum(self._even_groups(self.out_channels, self.split), dim=0):
                r2 = r2.item()
                num_neurons = r2 - r1

                # stack the input out_channels number of times
                y = torch.cat([x for _ in range(num_neurons)], dim=1)  

                y = y.permute(1,0,2,3)
                # each neuron applies its own shift
                y = randomshift(y, self.shifts[int(r1*self.in_channels):int(r2*self.in_channels)], self.learnable, self.max_shift, self.rounded_shifts, self.padding_mode)  
                y = y.permute(1,0,2,3)

                # x is N x (out_channels x in_channels) x H x W      10 x 12 x 28 x 28    ->     10 x 4 x 3 x 28 x 28    ->    10 x 4 x 6 x 28 x 28     ->   10 x 24 x 28 x 28
                # reshape to N x out_channels x in_channels x H x W  (N x in_channels x out_channels x H x W is incorrect)
                y = y.reshape(y.shape[0], num_neurons, self.in_channels, y.shape[-2], y.shape[-1])

                # add the qth dimension
                y = torch.cat([(y**i) for i in range(1, self.q + 1)], dim=2)

                # reshape to N x (out_channels x q x in_channels) x H x W
                y = y.reshape(y.shape[0], num_neurons * self.in_channels * self.q, y.shape[-2], y.shape[-1])  

                y = F.conv_transpose2d(y, self.weights[r1:r2].transpose(0,1).reshape(-1, 1, *self.kernel_size), bias=self.bias[r1:r2], padding=self.padding, stride=self.stride, dilation=self.dilation, output_padding=self.output_padding, groups=num_neurons) 
                ys.append(y)

                r1 = r2

            return torch.cat(ys, dim=1) 
        else:
            x = x.permute(1,0,2,3)
            x = randomshift(x, self.shifts, self.learnable, self.max_shift, self.rounded_shifts, self.padding_mode)
            x = x.permute(1,0,2,3)
            
            x = torch.cat([(x**i) for i in range(1, self.q+1)],dim=1)
            
            # Maybe wrong?
            x = F.conv_transpose2d(x, self.weights.transpose(0,1), bias=self.bias, padding=self.padding, stride=self.stride, dilation=self.dilation, output_padding=self.output_padding)        
            return x
        
    def _even_groups(self, n, g):
        g = max(min(n, g), 1)  # clamp n, 1
        rem = n % g
        numel = n // g
        groups = torch.cat((torch.tile(torch.tensor([numel + 1]), (rem,)), 
                            torch.tile(torch.tensor([numel]), (g - rem,))))
        return groups
    
    def reset_parameters(self):
        nn.init.uniform_(self.shifts, -self.max_shift, self.max_shift)
        with torch.no_grad():
            if self.rounded_shifts: self.shifts.data.round_()
                
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.weights, gain=gain)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)