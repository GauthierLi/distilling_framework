import torch 

from torch import nn

class DistillLoss(nn.Module):
    def __init__(self):
        super(DistillLoss, self).__init__()

    def forward(self, data):
        teacher_out, student_out = data['teacher'], data['student']
        import pdb; pdb.set_trace()
