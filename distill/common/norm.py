
# -*- encoding: utf-8 -*-
'''
@File    :   norm.py
@Time    :   2024/10/13 16:17:14
@Author  :   GauthierLi 
@Version :   1.0
@Contact :   lwklxh@163.com
@License :   Copyright (C) 2024 GauthierLi, All rights reserved.
'''

'''
This script include basic sparse norm module like NaiveSparseSyncBatchNorm1d, SparseSyncBatchNorm1d, etc.
'''


import functools
from collections import abc
from inspect import getfullargspec

import numpy as np
import torch
import torch.nn as nn
from torch import distributed as dist
from torch.autograd.function import Function
from torch.cuda.amp import autocast


class SparseTensor:
    def __init__(self, feats, coords, cur_tensor_stride=1):
        self.F = feats
        self.C = coords
        self.s = cur_tensor_stride
        self.coord_maps = {}
        self.kernel_maps = {}

    def check(self):
        if self.s not in self.coord_maps:
            self.coord_maps[self.s] = self.C

    def cuda(self):
        assert type(self.F) == torch.Tensor
        assert type(self.C) == torch.Tensor
        self.F = self.F.cuda()
        self.C = self.C.cuda()
        return self

    def detach(self):
        assert type(self.F) == torch.Tensor
        assert type(self.C) == torch.Tensor
        self.F = self.F.detach()
        self.C = self.C.detach()
        return self

    def to(self, device, non_blocking=True):
        assert type(self.F) == torch.Tensor
        assert type(self.C) == torch.Tensor
        self.F = self.F.to(device, non_blocking=non_blocking)
        self.C = self.C.to(device, non_blocking=non_blocking)
        return self

    def __add__(self, other):
        tensor = SparseTensor(self.F + other.F, self.C, self.s)
        tensor.coord_maps = self.coord_maps
        tensor.kernel_maps = self.kernel_maps
        return tensor



class AllReduce(Function):

    @staticmethod
    def forward(ctx, input):
        input_list = [
            torch.zeros_like(input) for k in range(dist.get_world_size())
        ]
        # Use allgather instead of allreduce in-place operations is unreliable
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


def force_fp32(apply_to=None, out_fp16=False):
    """Decorator to convert input arguments to fp32 in force.

    This decorator is useful when you write custom modules and want to support
    mixed precision training. If there are some inputs that must be processed
    in fp32 mode, then this decorator can handle it. If inputs arguments are
    fp16 tensors, they will be converted to fp32 automatically. Arguments other
    than fp16 tensors are ignored. If you are using PyTorch >= 1.6,
    torch.cuda.amp is used as the backend, otherwise, original mmcv
    implementation will be adopted.

    Args:
        apply_to (Iterable, optional): The argument names to be converted.
            `None` indicates all arguments.
        out_fp16 (bool): Whether to convert the output back to fp16.

    Example:

        >>> import torch.nn as nn
        >>> class MyModule1(nn.Module):
        >>>
        >>>     # Convert x and y to fp32
        >>>     @force_fp32()
        >>>     def loss(self, x, y):
        >>>         pass

        >>> import torch.nn as nn
        >>> class MyModule2(nn.Module):
        >>>
        >>>     # convert pred to fp32
        >>>     @force_fp32(apply_to=('pred', ))
        >>>     def post_process(self, pred, others):
        >>>         pass
    """

    def force_fp32_wrapper(old_func):

        @functools.wraps(old_func)
        def new_func(*args, **kwargs):
            # check if the module has set the attribute `fp16_enabled`, if not,
            # just fallback to the original method.
            if not isinstance(args[0], torch.nn.Module):
                raise TypeError('@force_fp32 can only be used to decorate the '
                                'method of nn.Module')
            if not (hasattr(args[0], 'fp16_enabled') and args[0].fp16_enabled):
                return old_func(*args, **kwargs)
            # get the arg spec of the decorated method
            args_info = getfullargspec(old_func)
            # get the argument names to be casted
            args_to_cast = args_info.args if apply_to is None else apply_to
            # convert the args that need to be processed
            new_args = []
            if args:
                arg_names = args_info.args[:len(args)]
                for i, arg_name in enumerate(arg_names):
                    if arg_name in args_to_cast:
                        new_args.append(
                            cast_tensor_type(args[i], torch.half, torch.float))
                    else:
                        new_args.append(args[i])
            # convert the kwargs that need to be processed
            new_kwargs = dict()
            if kwargs:
                for arg_name, arg_value in kwargs.items():
                    if arg_name in args_to_cast:
                        new_kwargs[arg_name] = cast_tensor_type(
                            arg_value, torch.half, torch.float)
                    else:
                        new_kwargs[arg_name] = arg_value
            # apply converted arguments to the decorated method
            with autocast(enabled=False):
                output = old_func(*new_args, **new_kwargs)
            # cast the results back to fp32 if necessary
            if out_fp16:
                output = cast_tensor_type(output, torch.float, torch.half)
            return output

        return new_func

    return force_fp32_wrapper


def cast_tensor_type(inputs, src_type, dst_type):
    """Recursively convert Tensor in inputs from src_type to dst_type.

    Args:
        inputs: Inputs that to be casted.
        src_type (torch.dtype): Source type..
        dst_type (torch.dtype): Destination type.

    Returns:
        The same type with inputs, but all contained Tensors have been cast.
    """
    if isinstance(inputs, nn.Module):
        return inputs
    elif isinstance(inputs, torch.Tensor):
        return inputs.to(dst_type)
    elif isinstance(inputs, str):
        return inputs
    elif isinstance(inputs, np.ndarray):
        return inputs
    elif isinstance(inputs, abc.Mapping):
        return type(inputs)({
            k: cast_tensor_type(v, src_type, dst_type)
            for k, v in inputs.items()
        })
    elif isinstance(inputs, abc.Iterable):
        return type(inputs)(
            cast_tensor_type(item, src_type, dst_type) for item in inputs)
    else:
        return inputs


class NaiveSparseSyncBatchNorm1d(nn.BatchNorm1d):
    """Syncronized Batch Normalization for 3D Tensors.
    Note:
        This implementation is modified from
        https://github.com/facebookresearch/detectron2/
        `torch.nn.SyncBatchNorm` has known unknown bugs.
        It produces significantly worse AP (and sometimes goes NaN)
        when the batch size on each worker is quite different
        (e.g., when scale augmentation is used).
        In 3D detection, different workers has points of different shapes,
        whish also cause instability.
        Use this implementation before `nn.SyncBatchNorm` is fixed.
        It is slower than `nn.SyncBatchNorm`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp16_enabled = False

    @force_fp32(out_fp16=True)
    def forward(self, input):
        assert input.dtype == torch.float32, \
            f'input should be in float32 type, got {input.dtype}'
        if not self.training or dist.get_world_size() == 1:
            return super().forward(input)
        assert input.shape[0] > 0, 'SyncBN does not support empty inputs'
        C = input.shape[1]
        mean = torch.mean(input, dim=[0])
        meansqr = torch.mean(input * input, dim=[0])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (
            mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1)
        bias = bias.reshape(1, -1)
        return input * scale + bias


class SparseSyncBatchNorm1d(nn.Module):
    def __init__(self,
                 num_features: int,
                 *,
                 eps: float = 1e-5,
                 momentum: float = 0.1) -> None:
        super().__init__()
        self.bn = NaiveSparseSyncBatchNorm1d(num_features=num_features, eps=eps, momentum=momentum)

    def forward(self, inputs):
        feats = inputs.F
        coords = inputs.C
        stride = inputs.s
        feats = self.bn(feats)
        outputs = SparseTensor(coords=coords, feats=feats, cur_tensor_stride=stride)
        outputs.coord_maps = inputs.coord_maps
        outputs.kernel_maps = inputs.kernel_maps
        return outputs


class NaiveSyncBatchNorm2d(nn.BatchNorm2d):
    """Syncronized Batch Normalization for 4D Tensors.
    Note:
        This implementation is modified from
        https://github.com/facebookresearch/detectron2/
        `torch.nn.SyncBatchNorm` has known unknown bugs.
        It produces significantly worse AP (and sometimes goes NaN)
        when the batch size on each worker is quite different
        (e.g., when scale augmentation is used).
        This phenomenon also occurs when the multi-modality feature fusion
        modules of multi-modality detectors use SyncBN.
        Use this implementation before `nn.SyncBatchNorm` is fixed.
        It is slower than `nn.SyncBatchNorm`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fp16_enabled = False

    @force_fp32(out_fp16=True)
    def forward(self, input):
        assert input.dtype == torch.float32, \
            f'input should be in float32 type, got {input.dtype}'
        if not self.training or dist.get_world_size() == 1:
            return super().forward(input)

        assert input.shape[0] > 0, 'SyncBN does not support empty inputs'
        C = input.shape[1]
        mean = torch.mean(input, dim=[0, 2, 3])
        meansqr = torch.mean(input * input, dim=[0, 2, 3])

        vec = torch.cat([mean, meansqr], dim=0)
        vec = AllReduce.apply(vec) * (1.0 / dist.get_world_size())

        mean, meansqr = torch.split(vec, C)
        var = meansqr - mean * mean
        self.running_mean += self.momentum * (
            mean.detach() - self.running_mean)
        self.running_var += self.momentum * (var.detach() - self.running_var)

        invstd = torch.rsqrt(var + self.eps)
        scale = self.weight * invstd
        bias = self.bias - mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return input * scale + bias


norm_cfg = {
        # format: layer_type: (abbreviation, module)
        'BN': ('bn', nn.BatchNorm2d),
        'BN1d': ('bn', nn.BatchNorm1d),
        'BN2d': ('bn', nn.BatchNorm2d),
        'BN3d': ('bn', nn.BatchNorm3d),
        'SyncBN': ('bn', nn.SyncBatchNorm),
        'GN': ('gn', nn.GroupNorm),
        'LN': ('ln', nn.LayerNorm),
        'IN': ('in', nn.InstanceNorm2d),
        'IN1d': ('in', nn.InstanceNorm1d),
        'IN2d': ('in', nn.InstanceNorm2d),
        'IN3d': ('in', nn.InstanceNorm3d),
        'NaiveSyncBN2d': ('bn', NaiveSyncBatchNorm2d),
        'NaiveSparseSyncBatchNorm1d': ('bn', NaiveSparseSyncBatchNorm1d),
        'SparseSyncBN1d': ('bn', SparseSyncBatchNorm1d),
        'NaiveSparseSyncBN1d': ('bn', NaiveSparseSyncBatchNorm1d),
    }


def build_norm_layer(cfg, num_features, postfix=''):
    """ Build normalization layer

    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.

    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in norm_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        abbr, norm_layer = norm_cfg[layer_type]
        if norm_layer is None:
            raise NotImplementedError

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN':
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer
