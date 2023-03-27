import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from superonn_final import SuperONN2d


def test(sonn: SuperONN2d, cnn: nn.Conv2d, xsonn=None, xcnn=None, device=None):
    cnn.weight.data = sonn.weight.data
    if sonn.bias is not None:
        cnn.bias.data = sonn.bias.data
    
    if xsonn is None:
        xsonn = torch.randn(2, sonn.in_channels, 32, 32, device=device)
    if xcnn is None:
        xcnn = xsonn

    y = cnn(xcnn)
    y_sonn = sonn(xsonn)
    return torch.allclose(y, y_sonn)


def test_all():
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
                                    sonn = SuperONN2d(in_channels, out_channels, kernel_size=kernel_size, q=q, padding=kernel_size//2, bias=bias, groups=groups, full_groups=full_groups, max_shift=max_shift, shift_init='zeros')
                                    cnn = nn.Conv2d(in_channels, cnn_out_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=bias, groups=sonn.groups)
                                    sonn.to(device)
                                    cnn.to(device)

                                    xsonn = torch.randn(1, in_channels, 32, 32)
                                    xsonn = xsonn.to(device)
                                    xcnn = torch.cat([torch.cat([xsonn**qpow for qpow in range(1, q+1)], dim=1) for _ in range(full_groups)], dim=1)
                                    xcnn = xcnn.to(device)

                                    assert test(sonn, cnn, xsonn, xcnn), f'test failed: {sonn}'
                                    print(f'Passed {i+1:>4d}/{tot:>4d}', end='\r')
                                    i += 1

if __name__ == '__main__':
    test_all()
    print('All tests passed')
