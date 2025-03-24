import os 
import torch 

from torch import nn
from typing import List
from .base_model import BaseModel

class Dinov2Backbone(BaseModel):
    """ 
    output multi-layer features
    """
    def __init__(self,
                 model_name: str = 'dinov2_vitg14_reg',
                 intermediate_layers: List[int] = [9, 19, 29, 39],
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dinov2_path = os.path.join(__file__, "../../../ext/dinov2")
        self.dinov2_model = torch.hub.load(os.path.abspath(dinov2_path), model_name, source="local")
        self.intermediate_layers = intermediate_layers

    def forward(self, x):
        return self.dinov2_model.get_intermediate_layers(x, self.intermediate_layers)

if __name__ == "__main__":
    with torch.no_grad():
        x = torch.rand((1, 3, 588, 966)).to('cuda')
        model = Dinov2Backbone().to('cuda')
        result = model(x) # get intermedia
