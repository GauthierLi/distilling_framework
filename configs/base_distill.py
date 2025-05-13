import os
import torch
import distill

from torch import nn
from torch import optim
from typing import List
from einops import rearrange
from torchvision import transforms
from distill.loss import DistillLoss
from distill.trainer import BaseTrainer
from distill.data.dataloader import PlayAndDebugDataloader
from distill.models.structure import DistillModel
from distill.models.backbone import Dinov2Backbone, LightClsBackbone

# ***************************************
#                Heads                   
# teacher heads
class VitHead(nn.Module):
    def __init__(
        self,
        input_shape: List[int],
        output_dim: int,
        intermediate_single_dim: int = 64,
        *args, **kwargs
    ):
        super(VitHead, self).__init__(*args, **kwargs)
        self.input_dim =input_shape 
        self.output_dim = output_dim
        self.intermediate_single_dim = intermediate_single_dim
        self.fc1 = nn.Linear(input_shape[1], intermediate_single_dim)
        self.fc2 = nn.Linear(input_shape[0], intermediate_single_dim)
        self.fc = nn.Sequential(
            nn.Linear(intermediate_single_dim**2, 1024),
            nn.Dropout(0.5),
            nn.Linear(1024, output_dim),
        )
        
    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = x.permute(0, 2, 1)
        x = self.fc2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ConvHead(VitHead):
    def __init__(
        self,
        input_shape: List[int],
        output_dim: int,
        intermediate_single_dim: int = 64,
        *args,
        **kwargs,
    ):
        super().__init__(
            [input_shape[0], input_shape[1] * input_shape[2]],
            output_dim,
            intermediate_single_dim,
            *args, **kwargs
        )

    def forward(self, x: torch.Tensor):
        x = rearrange(x, "b c h w -> b c (h w)")
        return super().forward(x)

# ****************************************************************
#                          required 

dataloader = PlayAndDebugDataloader(
    generate_img_size = {
        "teacher": (966, 588),
        "student": (960, 576)
    },
    transforms = [transforms.ToTensor()]
)

models = {
    "teacher": {
        "model": Dinov2Backbone(intermediate_layers=[9, 19, 29, 39]),
        "teacher_out_layers": {
            "indices": [0, 1, 2, 3],
            "shapes": [
                [1, 2898, 1536],
                [1, 2898, 1536],
                [1, 2898, 1536],
                [1, 2898, 1536],
            ],
            "heads":[
                VitHead(input_shape=[2898, 1536], output_dim=1024, intermediate_single_dim=64),
                VitHead(input_shape=[2898, 1536], output_dim=1024, intermediate_single_dim=64),
                VitHead(input_shape=[2898, 1536], output_dim=1024, intermediate_single_dim=64),
                VitHead(input_shape=[2898, 1536], output_dim=1024, intermediate_single_dim=64),
            ]
        },
    },
    "student": {
        "model": LightClsBackbone(cat_dim=0, in_channels=3, widths=[32, 64, 128, 256]),
        "student_out_layers": {
            "indices": [0, 1, 2, 3],
            "shapes": [
                [1, 32, 72, 120],
                [1, 64, 36, 60],
                [1, 128, 18, 30],
                [1, 256, 9, 15],
            ],
            "heads":[
                ConvHead(input_shape=[32, 72, 120], output_dim=1024, intermediate_single_dim=64),
                ConvHead(input_shape=[64, 36, 60], output_dim=1024, intermediate_single_dim=64),
                ConvHead(input_shape=[128, 18, 30], output_dim=1024, intermediate_single_dim=64),
                ConvHead(input_shape=[256, 9, 15], output_dim=1024, intermediate_single_dim=64),
            ]
        },
    },
}

loss = DistillLoss()
distill_structure = DistillModel
optimizer = optim.AdamW(
    [dict(params=models["student"]["model"].parameters())] +
    [dict(params=m.parameters(), lr=1e-4) for m \
            in models["student"]["student_out_layers"]["heads"]] +
    [dict(params=m.parameters(), lr=1e-4) for m \
            in models["teacher"]["teacher_out_layers"]["heads"]] ,
    lr=1e-3,
    weight_decay=1e-2,
)
scheduler = optim.lr_scheduler.StepLR(
    optimizer=optimizer,
    step_size=10,
    gamma=0.1
)

trainer = dict(
    type=BaseTrainer,
    device="cuda",
    epochs=100,
    save_period=1,
    monitor="off",
    early_stop=-1,
    validate_period=1,
    resume = False,
    max_saved = 3,
    save_dir = "workdirs/" + 
        f"{models['teacher']['model'].__class__.__name__}-{models['student']['model'].__class__.__name__}-distilling",
)

if __name__ == "__main__":
    for ds in dataloader:
        import pdb; pdb.set_trace()
