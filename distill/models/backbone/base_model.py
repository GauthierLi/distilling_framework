from torch import nn
from typing import Literal


class BaseModel(nn.Module):
    def __init__(self,
                 model_structure: Literal['cnn', 'transformer'] = 'cnn',
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_structure = model_structure
