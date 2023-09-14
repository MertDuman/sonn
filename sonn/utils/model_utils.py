import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as AG
import copy
import numpy as np
import matplotlib.pyplot as plt


def get_num_params(model):
    ''' Counts all parameters in a model. '''
    return sum(p.numel() for p in model.parameters())


def print_model_param_info(model, cell_len=30, cell_algn="<", in_filters=None, ex_filters=None, ex_zero=False, ex_container=False, clip=True):
    ''' Prints model parameter information. '''
    in_filters = None if in_filters is None else set(in_filters)
    ex_filters = None if ex_filters is None else set(ex_filters)
    table_width = cell_len * 3 + 2

    # First pass to calculate total params
    total_params = 0
    total_layers = 0
    for n, m in model.named_modules():
        m_params = get_num_params(m)
        c_params = sum(get_num_params(c) for c in m.children())
        param_diff = m_params - c_params

        # Is container if it has no parameters but has children
        is_container = param_diff == 0 and c_params != 0
        
        if ex_container and is_container:
            continue

        if ex_zero and m_params == 0:
            continue

        type_name = type(m).__name__
        if in_filters is not None and (type_name not in in_filters and type(m) not in in_filters):
            continue
        if ex_filters is not None and (type_name in ex_filters or type(m) in ex_filters):
            continue
            
        total_params += param_diff
        total_layers += 1

    # Second pass to print info
    has_seperator = False
    separated_now = False
    print(f'{"Layer":{cell_algn}{cell_len}} {"Structure":{cell_algn}{cell_len}} {"Params":{cell_algn}{cell_len}}')
    print(f"{type(model).__name__:{cell_algn}{cell_len}} {total_layers:{cell_algn}{cell_len}} {total_params:{cell_algn}{cell_len}}")
    for n, m in model.named_modules():
        separated_now = False

        m_params = get_num_params(m)
        c_params = sum(get_num_params(c) for c in m.children())
        param_diff = m_params - c_params

        # Is container if it has no parameters but has children
        is_container = param_diff == 0 and c_params != 0

        # Print separator if no dot in name and we haven't printed a separator yet
        if n.count('.') == 0 and not has_seperator:
            print("-" * table_width)
            has_seperator = True
            separated_now = True

        if ex_container and param_diff == 0 and m_params != 0:
            continue

        if ex_zero and m_params == 0: 
            continue

        type_name = type(m).__name__
        if in_filters is not None and (type_name not in in_filters and type(m) not in in_filters):
            continue
        if ex_filters is not None and (type_name in ex_filters or type(m) in ex_filters):
            continue

        # If we passed all filters, and this is not the iteration that printed a separator, we no longer have a separator
        if not separated_now:
            has_seperator = False
        
        if clip and len(n) >= cell_len:
            n = n[:cell_len-3] + '...'
        if clip and len(type_name) >= cell_len: 
            type_name = type_name[:cell_len-3] + '...'

        # Containers
        final_params = param_diff
        if is_container:
            final_params = f"{m_params} - C"
        print(f"{type_name:{cell_algn}{cell_len}} {n:{cell_algn}{cell_len}} {final_params:{cell_algn}{cell_len}}")


def get_receptive_field(x, model, target_idx=None, absnorm=False):
    # Input requires grad
    x = x.requires_grad_(True)
    # y has grad_fn and requires_grad
    y = model(x)
    if target_idx is not None:
        y = y[target_idx]
    y = y.view(y.shape[0], -1)
    grd_center = y.shape[-1] // 2 + x.shape[-1] // 2
    grd = torch.zeros_like(y)
    grd[:, grd_center] = 1
    # grad is calculated, but model weights are untouched, their grad fields are not filled, and y's computation graph is freed
    grad = AG.grad(y, x, grd)[0]
    if absnorm:
        grad = torch.abs(grad)
        grad = (grad - grad.amin(dim=(2,3), keepdim=True)) / (grad.amax(dim=(2,3), keepdim=True) - grad.amin(dim=(2,3), keepdim=True))
    return grad


def plot_receptive_fields(*models, x):
    num_models = len(models)
    rows, cols = [(r, num_models // r) for r in range(1, int(np.sqrt(num_models)) + 1) if num_models % r == 0][-1]
    if rows == 1:
        rows = int(np.sqrt(num_models))
        cols = num_models // rows + 1
    fig, axs = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows), squeeze=False)

    for i in range(rows * cols):
        if i >= num_models:
            fig.delaxes(axs[i // cols, i % cols])
            continue
        model = models[i]
        rf = get_receptive_field(x, model).abs()
        rf = (rf - rf.amin(dim=(2,3), keepdim=True)) / (rf.amax(dim=(2,3), keepdim=True) - rf.amin(dim=(2,3), keepdim=True))
        ax = axs[i // cols, i % cols]
        ax.imshow(rf[0].permute(1, 2, 0).cpu())
        ax.set_title(type(model).__name__)

    fig.tight_layout()
    plt.subplots_adjust(wspace=-0.55, hspace=0.15)
    plt.show()