import torch 

from torch import nn
import torch.nn.functional as F

class DistillLoss(nn.Module):
    def __init__(self):
        super(DistillLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, data):
        teacher_out, student_out = data['teacher'], data['student']
        loss = 0.
        for tea, stu in zip(teacher_out['out'], student_out['out']):
            loss += self.ce_loss(F.softmax(stu), F.softmax(tea))
        return loss / len(teacher_out['out'])
