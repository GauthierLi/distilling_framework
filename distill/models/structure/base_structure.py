import torch 

from torch import nn
from typing import Literal
from torch.optim import Optimizer
from abc import abstractmethod, ABCMeta
from distill.utils import batch_call

class BaseStructure(nn.Module):
    def __init__(self,
                 teacher: dict,
                 student: dict,
                 loss: nn.Module,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher["model"]
        self.student_model = student["model"]
        self.teacher_layers = teacher["teacher_out_layers"]
        self.student_layers = student["student_out_layers"]
        self.loss = loss

    def _train_step(self, x, *args, **kwargs):
        output = self._forward(x)
        loss = self.loss(output)
        return loss

    @abstractmethod
    def _val_step(self, data, *args, **kwargs):
        raise NotImplementedError

    def _forward(self, data, *args, **kwargs):
        # teacher out
        teacher_in = data
        student_in = data
        if isinstance(data, dict):
            teacher_in = data['teacher']
            student_in = data['student']
        if isinstance(data, list):
            teacher_in = data[0]
            student_in = data[1]

        with torch.no_grad():
            teacher_out = self.teacher_model(teacher_in)
        # student out
        student_out = self.student_model(student_in)
        t_feat_out = []
        s_feat_out = []

        for i, feat in enumerate(teacher_out):
            if i in self.teacher_layers['indices']:
                t_feat_out.append(feat)

        for i, feat in enumerate(student_out):
            if i in self.student_layers['indices']:
                s_feat_out.append(feat)

        t_out = batch_call(
            self.teacher_layers['heads'],
            [[f] for f in t_feat_out]
        )

        s_out = batch_call(
            self.student_layers['heads'],
            [[f] for f in s_feat_out]
        )

        self.teacher_layers['out'] = t_out
        self.student_layers['out'] = s_out

        self.teacher_layers['feats'] = t_out
        self.student_layers['feats'] = s_out
        output = dict(
            teacher=self.teacher_layers,
            student=self.student_layers
        )
        return output

    def forward(self,
                data: torch.Tensor,
                mode: Literal["train", "val", "norm"],
                *args, **kwargs):
        if mode == "train":
            return self._train_step(data, *args, **kwargs)
        if mode == "val":
            return self._val_step(data, *args, **kwargs)
        if mode == "norm":
            return self._forward(data, *args, **kwargs)

