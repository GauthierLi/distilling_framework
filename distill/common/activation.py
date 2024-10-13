# -*- encoding: utf-8 -*-
'''
@File    :   activation.py
@Time    :   2024/10/13 16:16:57
@Author  :   GauthierLi 
@Version :   1.0
@Contact :   lwklxh@163.com
@License :   Copyright (C) 2024 GauthierLi, All rights reserved.
'''

'''
Description here ...
'''

import torch
import torch.nn as nn

__all__ = ["SiLU", "get_activation"]


def get_activation(name="relu", inplace=True):
    """A function that can easily get activation function.

    Args:
        name (string): relu, silu, lrelu，
            elu, relu6, rrelu, selu, celu,
            gelu, hardshrink, sigmoid, tanh,
            hardsigmoid, hardwish

        inplace (int): Same as nn.ReLU
    Returns:
        Tensor: nn.Module

    """
    name = name.lower()
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == "elu":
        module = nn.ELU(1.0, inplace=inplace)
    elif name == "relu6":
        module = nn.ReLU6(inplace=inplace)
    elif name == "rrelu":
        module = nn.RReLU(inplace=inplace)
    elif name == "selu":
        module = nn.SELU(inplace=inplace)
    elif name == "celu":
        module = nn.CELU(1.0, inplace=inplace)
    elif name == "gelu":
        module = nn.GELU()
    elif name == "hardshrink":
        module = nn.Hardshrink(0.5)
    elif name == "sigmoid":
        module = nn.Sigmoid()
    elif name == "tanh":
        module = nn.Tanh()
    elif name == "hardsigmoid":
        module = nn.Hardsigmoid(inplace=inplace)
    elif name == "hardwish":
        module = nn.Hardswish(inplace=inplace)
    elif name == "identity":
        module = nn.Identity()

    else:
        raise AttributeError("Unsupported activation type: {}".format(name))
    return module


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU().
    downward compatible with Deeproute.ai Inference Engine.
    """

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
