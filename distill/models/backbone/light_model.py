import torch
import torch.nn as nn
from base_model import BaseModel

from typing import Sequence, Literal
from distill.loger import gaulog
from distill.common import ConvBnAct, LightResBlock

class LightClsBackbone(BaseModel):
    stage_names = ['down_sample', 'stage1', 'stage2', 'stage3']
    stage_conv_params = {
        'down_sample': dict(kernel_size=9, stride=8),
        'stage1': dict(kernel_size=3, stride=2),
        'stage2': dict(kernel_size=3, stride=2),
        'stage3': dict(kernel_size=3, stride=2),
        # 'stage4': dict(kernel_size=3, stride=2),
    }

    def __init__(
        self,
        cat_dim=0,
        in_channels=3,
        widths=[32, 64, 128, 256],
        model_structure: Literal['cnn', 'transformer']="cnn",
        init_cfg=None,
    ):
        super().__init__(model_structure=model_structure)
        self.cat_dim = cat_dim
        self.wid_stages = widths

        stage_in_channels = in_channels
        self.module_list = []
        for stage_name, width in zip(self.stage_names, widths):
            conv_param = self.stage_conv_params[stage_name]
            conv = ConvBnAct(stage_in_channels, width, **conv_param)
            if stage_name == 'down_sample':
                setattr(self, stage_name, conv)
            else:
                blocks = [
                conv,
                LightResBlock(width, use_ffn=False),
                LightResBlock(width, use_ffn=False),
                ]
                setattr(self, stage_name, nn.Sequential(*blocks))
            stage_in_channels = width
            self.module_list.append(stage_name)
        self._last_channel = widths[-1]

    def forward(self, x):
        out = []
        for i, layer_name in enumerate(self.module_list):
            light_layer = getattr(self, layer_name)
            x = light_layer(x)
            out.append(x)
        return out

    def __str__(self) -> str:
        print_str = "\n# ===================== Total Layer ====================== #\n"
        str_num = len(print_str)
        print_str += f"Layer number: {len(self.stage_conv_params)}".center(str_num)
        for name, v in self.stage_conv_params.items():
            print_str += "\n"
            print_str += f"{name}: {v}".center(str_num)
        return print_str
    
if __name__ == "__main__":
    x = torch.rand((1,3,576, 960)).to("cuda")
    model = LightClsBackbone().to("cuda")
    gaulog.debug(model)
    out = model(x)
    gaulog.info("Out shape: " +  str(len(out)) + f",\nshape 1: , {out[0].shape}, \nshape -1: {out[-1].shape}")
    import pdb; pdb.set_trace()
