import os
import distill

from torch import optim
from torchvision import transforms
from distill.data.dataloader import PlayAndDebugDataset
from distill.loss import DistillLoss
from distill.models.structure import DistillModel
from distill.models.backbone import Dinov2Backbone, LightClsBackbone

dataset = PlayAndDebugDataset(
    generate_img_size = (960, 576),
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
        },
    },
}

loss = DistillLoss()
distill_structure = DistillModel
optimizer = optim.AdamW(
    params=models["student"]["model"].parameters(),
    lr=1e-3, 
    weight_decay=1e-2
)

trainer = dict(
    epochs=100,
    save_period=1,
    monitor="off",
    early_stop=-1,
    train=[("train", 1), ("val", 1)],
    resume = False,
    save_dir = "workdirs/" + f"{models['teacher']}-{models['student']}-distilling",
    log_dir = os.path.join("workdirs/" + f"{models['teacher']}-{models['student']}-distilling", "logs")
)

if __name__ == "__main__":
    for ds in dataloader:
        import pdb; pdb.set_trace()
