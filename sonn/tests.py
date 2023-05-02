import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from superonn_final import SuperONN2d, take_qth_power


def test(net1: SuperONN2d, net2: nn.Conv2d, x_net1=None, x_net2=None, device=None):
    net2.weight.data = net1.weight.data
    if net1.bias is not None:
        net2.bias.data = net1.bias.data
    
    if x_net1 is None:
        x_net1 = torch.randn(2, net1.in_channels, 32, 32, device=device)
    if x_net2 is None:
        x_net2 = x_net1

    y_net2 = net2(x_net2)
    y_net1 = net1(x_net1)
    return torch.allclose(y_net2, y_net1)


def test_cnn_likeness():
    test_values = {
        'in_channels': [6, 12],
        'out_channels': [12, 24],
        'kernel_size': [1,2,3,4,5],
        'bias': [False, True],
        'full_groups': [1, 2, 3],
        'groups': [1, 6],
        'max_shift': [0, 5],
        'q': [1, 2, 3, 4, 5],
    }

    i = 0
    tot = np.prod([len(test_values[key]) for key in test_values])
    device = 'cpu'# torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for in_channels in test_values['in_channels']:
        for out_channels in test_values['out_channels']:
            for kernel_size in test_values['kernel_size']:
                for bias in test_values['bias']:
                    for full_groups in test_values['full_groups']:
                        for groups in test_values['groups']:
                            for max_shift in test_values['max_shift']:
                                for q in test_values['q']:

                                    cnn_out_channels = out_channels * full_groups * q  # this many channels will be created by the SuperONN2d
                                    sonn = SuperONN2d(in_channels, out_channels, kernel_size=kernel_size, q=q, padding=kernel_size//2, bias=bias, groups=groups, full_groups=full_groups, max_shift=max_shift, shift_init='zeros', with_w0=True)
                                    cnn = nn.Conv2d(in_channels, cnn_out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=bias, groups=sonn.groups)
                                    sonn.to(device)
                                    cnn.to(device)

                                    xsonn = torch.randn(2, in_channels, 32, 32)
                                    xsonn = xsonn.to(device)
                                    xcnn = torch.cat([take_qth_power(xsonn, q, with_w0=True) for _ in range(full_groups)], dim=1)
                                    #xcnn = torch.cat([torch.cat([xsonn**qpow for qpow in range(1, q+1)], dim=1) for _ in range(full_groups)], dim=1)
                                    xcnn = xcnn.to(device)

                                    assert test(sonn, cnn, xsonn, xcnn), f'test failed: {sonn}'
                                    print(f'Passed {i+1:>4d}/{tot:>4d}', end='\r')
                                    i += 1

if __name__ == '__main__':
    test_cnn_likeness()
    print('All tests passed')








## Unrelated
def num_nonneg_X_that_sum_to_Y(X, Y):
    ''' 
    Balls and bins problem. How many ways can you distribute Y identical balls into X bins?
    Ball: o
    Bin separator:  |
    Four balls to three bins:
        One possibility:        o o | o | o
        Another possibility:    o | o o | o
    It is the repeating permutations of 4 o's and 2 |'s: 6! / (4! * 2!) in other words comb(6, 2)
    '''
    import math
    return math.comb(Y+X-1, X-1)

def num_pos_X_that_sum_to_Y(X, Y):
    ''' 
    Balls and bins problem, but we first distribute 1 ball to each bin so they can't be empty.
    '''
    import math
    return math.comb((Y-X)+X-1, X-1)

def get_nonneg_X_that_sum_to_Y(X, Y):
    bins = np.zeros(X)
    np.add.at(bins, np.random.randint(0, X, Y), 1)
    return bins

def get_pos_X_that_sum_to_Y(X, Y):
    bins = np.ones(X)
    np.add.at(bins, np.random.randint(0, X, Y - X), 1)
    return bins