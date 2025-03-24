import torch 
import numpy as np

from .base_data import BaseDataset
from torch.utils.data import DataLoader, default_collate
from typing import Any, List, Callable, Tuple

class PlayAndDebugDataset(BaseDataset):
    def __init__(self, 
                 transforms:List[Callable] = None, 
                 max_try: int = 5,
                 fake_datasize: int = 100,
                 generate_img_size: Tuple[int, int] = (960, 576),
                 *args, 
                 **kwargs):
        self.fake_datasize = fake_datasize
        self.generate_img_size = generate_img_size
        super(PlayAndDebugDataset, self).__init__(transforms, max_try, *args, **kwargs)

    def _get_dataset_list(self):
        return [0] * self.fake_datasize

    def _parse_data(self, data: Any):
        img = np.random.randint(0, 255, size=(self.generate_img_size[0], self.generate_img_size[1], 3))
        return img

class PlayAndDebugDataloader(DataLoader):
    def __init__(self, 
                 transforms:List[Callable] = None, 
                 max_try: int = 5,
                 fake_datasize: int = 100,
                 generate_img_size: Tuple[int, int] = (960, 576),
                 batch_size: int = 1,
                 shuffle: bool = False,
                 pin_memory: bool = False,
                 collate_fn: Callable = default_collate,
                 *args, 
                 **kwargs):
        dataset = PlayAndDebugDataset(transforms=transforms,
                                      max_try=max_try,
                                      fake_datasize=fake_datasize,
                                      generate_img_size=generate_img_size)
        super(PlayAndDebugDataloader, self).__init__(dataset,
                                                     batch_size=batch_size,
                                                     shuffle=shuffle,
                                                     pin_memory=pin_memory,
                                                     collate_fn=collate_fn,
                                                     *args, **kwargs)
