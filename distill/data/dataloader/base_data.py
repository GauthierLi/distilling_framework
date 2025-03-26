import torch 
import traceback

from typing import Any, List, Callable
from abc import abstractmethod, ABCMeta
from torch.utils.data import DataLoader, Dataset

class BaseDataset(Dataset):
    def __init__(self,
                 transforms:List[Callable] = None, 
                 max_try: int = 5,
                 *args, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.max_try = max_try
        self.transforms = transforms
        self.data_info = self._get_dataset_list()

    @abstractmethod
    def _get_dataset_list(self):
        raise NotImplementedError

    @abstractmethod
    def _parse_data(self, data: Any):
        raise NotImplementedError

    def process_data(self, idx: int):
        data_meta = self.data_info[idx]
        data = self._parse_data(data_meta)
        if self.transforms:
            for trans in self.transforms:
                data = trans(data)
        return data

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, 
                 idx: int, 
                 *args: Any, 
                 **kwds: Any) -> Any:
        try_time = 0
        while try_time < self.max_try:
            try:
                return self.process_data(idx)
            except Exception as e:
                try_time += 1
                print("="*35)
                print(f"Error: {e}, try {try_time + 1}/{self.max_try} time")
                traceback.print_exc()
                error = traceback.format_exc()
                print(error)

