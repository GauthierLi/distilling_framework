import torch 
from distill.loger import gaulog

from torch import nn
from torch.optim import Optimizer
from .base_structure import BaseStructure

class DistillModel(BaseStructure):
    def __init__(self,
                 teacher: dict,
                 student: dict,
                 loss: nn.Module):
        super(DistillModel, self).__init__(teacher, student, loss)


