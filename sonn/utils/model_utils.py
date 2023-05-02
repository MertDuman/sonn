import torch
import torch.nn as nn
import torch.nn.functional as F


def get_num_params(model):
    ''' Counts all parameters in a model. '''
    num_params = 0
    for param in model.parameters():
        num_params += torch.prod(torch.tensor(param.shape)).item()
    return num_params