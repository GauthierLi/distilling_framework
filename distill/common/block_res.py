""" This script will include basic blokcs used in different Network """

import torch.nn as nn

from .conv_modules import ConvBn, ConvBnAct, get_activation


class ResBasicBlock(nn.Module):
    """
    To Optimize CSPNet ZY custom build ResBasicBlock. Plz not use it to build offical resnet !!!
    -------------------------------
    forward flow:
        raw = x
        x = conv1(x, stride=stride, k=3)
        x = conv2(x, stride=1, k=3)
        x = acti(bn(x + conv3(raw, stride=stride, k=1)))

    """

    def __init__(self, in_channels, out_channels, stride, act="silu"):
        super().__init__()
        self.conv1 = ConvBnAct(in_channels, out_channels, 3, stride=stride, act=act)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act)

    def forward(self, x):
        raw = x
        x = self.conv1(x)
        x = self.bn(self.conv2(x))
        sc = self.conv3(raw)
        x = self.act(sc + x)
        return x


class ResBasicStage(nn.Module):
    # composed of multi-ResBasicBlock, keep same kwargs as `CSPLayer`
    def __init__(self, in_channels, out_channels, stride, n, act="silu"):
        super().__init__()
        self.stages = nn.ModuleList()
        for i in range(n):
            self.stages.append(
                ResBasicBlock(in_channels, out_channels, stride, act=act)
                if i == 0
                else ResBasicBlock(out_channels, out_channels, 1, act=act)
            )

    def forward(self, x):
        for block in self.stages:
            x = block(x)
        return x


class FFN(nn.Module):
    """
    a powerful trick to improve light network.

    """
    def __init__(self, in_channels, out_channels, norm_cfg=None):
        super().__init__()
        self.ffn = nn.Sequential(
            ConvBnAct(in_channels, out_channels*2, 1, 1, padding=0, norm_cfg=norm_cfg),
            ConvBn(out_channels*2, out_channels, 1, 1,  padding=0, norm_cfg=norm_cfg)
        )

    def forward(self, x):
        identity = x
        x = self.ffn(x)
        x = x + identity
        return x


class LightResBlock(nn.Module):
    """
    a light res basic block

    """
    def __init__(self, in_channels, out_channels=None, use_ffn=False, inplace=True, norm_cfg=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = nn.Sequential(
            ConvBnAct(in_channels, in_channels, (1, 3), 1, padding=(0, 1), norm_cfg=norm_cfg),
            ConvBnAct(in_channels, out_channels, 3, 1, padding=1, norm_cfg=norm_cfg),
            ConvBn(out_channels, out_channels, (3, 1), 1, padding=(1, 0), norm_cfg=norm_cfg)
        )
        self.relu = nn.ReLU(inplace=inplace)
        self.use_ffn = use_ffn
        if use_ffn:
            self.ffn = FFN(out_channels, out_channels, norm_cfg=norm_cfg)

    def forward(self, input):
        x = self.conv(input)
        x = self.relu(x + input)
        if self.use_ffn:
            x = self.ffn(x)
        return x


def replace_feature(out, new_features):
    out.F = new_features
    return out
