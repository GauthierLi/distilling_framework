from torch import nn

from distill.common.sparse_tensor.sparse_tensor import *

from ..functional import *


class GlobalAveragePooling(nn.Module):
    def forward(self, inputs):
        return global_avg_pool(inputs)


class GlobalMaxPooling(nn.Module):
    def forward(self, inputs):
        return global_max_pool(inputs)
