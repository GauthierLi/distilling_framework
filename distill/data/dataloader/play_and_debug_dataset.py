import torch
import numpy as np

from .base_data import BaseDataset
from torch.utils.data import DataLoader, default_collate
from typing import Any, List, Callable, Tuple, Dict


class PlayAndDebugDataset(BaseDataset):
    def __init__(
        self,
        transforms: List[Callable] = None,
        max_try: int = 5,
        fake_datasize: int = 100,
        generate_img_size: Dict[str, Tuple[int, int]] = {
            "teacher": (966, 588),
            "student": (960, 576),
        },
        *args,
        **kwargs,
    ):
        self.fake_datasize = fake_datasize
        self.generate_img_size = generate_img_size
        super(PlayAndDebugDataset, self).__init__(transforms, max_try, *args, **kwargs)

    def _get_dataset_list(self):
        return [0] * self.fake_datasize

    def _parse_data(self, data: Any):
        data = dict()
        for key in self.generate_img_size:
            data[key] = np.random.randint(
                0,
                255,
                size=(
                    self.generate_img_size[key][0],
                    self.generate_img_size[key][1],
                    3,
                ),
            ).astype("uint8")
        return data

    def process_data(self, idx: int):
        data_meta = self.data_info[idx]
        data = self._parse_data(data_meta)
        if self.transforms:
            for trans in self.transforms:
                for key in data:
                    data[key] = trans(data[key])
        return data


class PlayAndDebugDataloader(DataLoader):
    def __init__(
        self,
        transforms: List[Callable] = None,
        max_try: int = 5,
        fake_datasize: int = 100,
        generate_img_size: Dict[str, Tuple[int, int]] = {
            "teacher": (966, 588),
            "student": (960, 576),
        },
        batch_size: int = 1,
        shuffle: bool = False,
        pin_memory: bool = False,
        collate_fn: Callable = default_collate,
        *args,
        **kwargs,
    ):
        self.dataset = PlayAndDebugDataset(
            transforms=transforms,
            max_try=max_try,
            fake_datasize=fake_datasize,
            generate_img_size=generate_img_size,
        )
        super(PlayAndDebugDataloader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            *args,
            **kwargs,
        )
