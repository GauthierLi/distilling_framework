""" This script include basic conv module like ConvBNAct, SeparableConv, etc."""
# Copyright (c) Deeproute.ai. All rights reserved.

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
# import drcv.ops.torch_sparse.nn as spnn

from distill.common.weight_init import normal_init as _normal_init

from .activation import get_activation
from .norm import build_norm_layer

__all__ = [
    "Scale",
    "ConvBnAct",
    "Focus",
    "SeparableConv2D",
    "ConvBn",
]


class Focus(nn.Module):
    """
    Focus width and height information into channel space.
    About this Layer: https://github.com/ultralytics/yolov5/discussions/3181m1

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d. Default: 1.
        act (string): relu, rrelu, silu...
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, act="silu"):
        super().__init__()
        self.conv = ConvBnAct(
            in_channels * 4, out_channels, kernel_size, stride, act=act
        )

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]

        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )

        return self.conv(x)


class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale


class ConvBnAct(nn.Module):
    """A conv block that bundles conv/norm/activation layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool): Same as nn.Conv2d.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        act="relu",
        inplace=True,
        padding=None,
        norm_cfg=None,
    ):
        super(ConvBnAct, self).__init__()
        if norm_cfg is None:
            norm_cfg = dict(type="BN2d")
        if padding is None:
            padding = int((kernel_size - 1) / 2)
        else:
            padding = padding
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )
        self.bn = build_norm_layer(norm_cfg, out_channels)[1]
        self.act = get_activation(act, inplace)

    def forward(self, x):
        x = self.conv(x)  # implicit_convolve_sgemm
        x = self.bn(x)  # vectorized_elementwise_kernel bn_fw_tr_1C11_kernel_NCHW
        x = self.act(x)
        return x

    def fuseforward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class ConvBn(nn.Module):
    """A conv block that bundles conv/norm layers.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int | tuple[int]): Same as nn.Conv2d.
        stride (int | tuple[int]): Same as nn.Conv2d.
        padding (int | tuple[int]): Same as nn.Conv2d.
        groups (int): Same as nn.Conv2d.
        bias (bool): Same as nn.Conv2d.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        padding=None,
        norm_cfg=None,
    ):
        super(ConvBn, self).__init__()
        if norm_cfg is None:
            norm_cfg = dict(type="BN2d")
        if padding is None:
            padding = int((kernel_size - 1) / 2)
        else:
            padding = padding
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )
        self.bn = build_norm_layer(norm_cfg, out_channels)[1]

    def forward(self, x):
        x = self.conv(x)  # implicit_convolve_sgemm
        x = self.bn(x)  # vectorized_elementwise_kernel bn_fw_tr_1C11_kernel_NCHW
        return x


class SeparableConv2D(nn.Module):
    """Depthwise separable convolution module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        kernel_size (int or tuple[int]): Same as nn.Conv2d.
        stride (int or tuple[int]): Same as nn.Conv2d. Default: 1.
        dw_conv (nn.Conv2d): Depthwise conv.
            If specific, will not create inside `SeparableConv2D`.
        pw_conv (nn.Conv2d): Pointwise conv.
            If specific, will not create inside `SeparableConv2D`.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dw_conv: Optional[nn.Conv2d] = None,
        pw_conv: Optional[nn.Conv2d] = None,
        act="relu",
        norm_cfg=None,
    ):
        super(SeparableConv2D, self).__init__()
        if norm_cfg is None:
            norm_cfg = dict(type="BN2d")
        self.in_channels = in_channels
        self.out_channels = out_channels
        padding = int((kernel_size - 1) / 2)

        self.depthwise = (
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
                groups=in_channels,
                bias=False,
            )
            if dw_conv is None
            else dw_conv
        )

        self.pointwise = (
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)
            if pw_conv is None
            else pw_conv
        )

        self.bn = build_norm_layer(norm_cfg, out_channels)[1]
        self.act = get_activation(act)
        self._init_weights()

    def _init_weights(self):
        _normal_init(self.depthwise, std=np.sqrt(1.0 / self.in_channels))
        _normal_init(self.pointwise, std=np.sqrt(1.0 / self.out_channels), bias=0)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def fuseforward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.act(x)
        return x

