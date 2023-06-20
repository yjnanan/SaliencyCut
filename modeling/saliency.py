import math
import numpy as np
import torch
import torch.nn.functional as F

def bspline_kernel_2d(sigma=[1, 1], order=2, asTensor=False, dtype=torch.float32, device='cuda'):
    '''
    generate bspline 2D kernel matrix.
    From wiki: https://en.wikipedia.org/wiki/B-spline, Fast b-spline interpolation on a uniform sample domain can be
    done by iterative mean-filtering
    :param sigma: tuple integers, control smoothness
    :param order: the order of interpolation
    :param asTensor:
    :param dtype: data type
    :param use_gpu: bool
    :return:
    '''
    kernel_ones = torch.ones(1, 1, *sigma)
    kernel = kernel_ones
    padding = np.array(sigma)

    for i in range(1, order + 1):
        kernel = F.conv2d(kernel, kernel_ones, padding=(i * padding).tolist()) / ((sigma[0] * sigma[1]))

    if asTensor:
        return kernel.to(dtype=dtype, device=device)
    else:
        return kernel.numpy()

def get_bspline_kernel(spacing=[32, 32], order=3):
    '''
    :param order init: bspline order, default to 3
    :param spacing tuple of int: spacing between control points along h and w.
    :return:  kernel matrix
    '''
    _kernel = bspline_kernel_2d(spacing, order=order, asTensor=True)
    _padding = (np.array(_kernel.size()[2:]) - 1) / 2
    _padding = _padding.astype(dtype=int).tolist()
    return _kernel, _padding

def rescale_intensity(data, new_min=0, new_max=1, group=4, eps=1e-20):
    '''
    rescale pytorch batch data
    :param data: N*1*H*W
    :return: data with intensity ranging from 0 to 1
    '''
    bs, c, h, w = data.size(0), data.size(1), data.size(2), data.size(3)
    data = data.view(bs * c, -1)
    # pytorch 1.3
    old_max = torch.max(data, dim=1, keepdim=True).values
    old_min = torch.min(data, dim=1, keepdim=True).values

    new_data = (data - old_min + eps) / (old_max - old_min + eps) * (new_max - new_min) + new_min
    new_data = new_data.view(bs, c, h, w)
    return new_data