import torch 

from torch import nn
from typing import Literal
from torch.optim import Optimizer
from abc import abstractmethod, ABCMeta

class BaseStructure(nn.Module):
    def __init__(self,
                 teacher: dict,
                 student: dict,
                 optimizer: Optimizer,
                 loss: nn.Module,
                 *args,
                 **kwargs) -> None:
        self.teacher_model = teacher["model"]
        self.student_model = student["model"]
        self.teacher_layers = teacher["teacher_out_layers"]
        self.student_layers = student["student_out_layers"]
        self.loss = loss
        self.optimizer = optimizer
        super().__init__(*args, **kwargs)

    def _train_step(self, x, *args, **kwargs):
        output = self._forward(x)
        student_out = output['student']["out"]
        teacher_out = output['teacher']["out"]

    @abstractmethod
    def _val_step(self, data: dict, *args, **kwargs):
        raise NotImplementedError

    def _forward(self, x, *args, **kwargs):
        # teacher out
        with torch.no_grad():
            teacher_out = self.teacher_model(x)
        # student out
        student_out = self.student_model(x)
        t_out = []
        s_out = []

        for i, feat in enumerate(teacher_out):
            if i in self.teacher_layers['indices']:
                t_out.append(feat)

        for i, feat in enumerate(student_out):
            if i in self.student_layers['indices']:
                s_out.append(feat)

        self.teacher_layers['out'] = t_out
        self.student_layers['out'] = s_out
        output = dict(
            teacher=self.teacher_layers,
            student=self.student_layers
        )
        return output

    def forward(self,
                data: dict,
                mode: Literal["train", "val", "norm"],
                *args, **kwargs):
        if mode == "train":
            return self._train_step(data, *args, **kwargs)
        if mode == "val":
            return self._val_step(data, *args, **kwargs)
        if mode == "norm":
            return self._forward(data, *args, **kwargs)

