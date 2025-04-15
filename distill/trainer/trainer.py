import torch 

from tqdm import tqdm 
from typing import Dict, Any, Iterable
from distill.loger import gaulog

class BaseTrainer(object):
    def __init__(self,
                 device: str,
                 epochs: int,
                 models: Dict[str, Dict[str, Any]],
                 optimizer: torch.optim.Optimizer,
                 loss: torch.nn.Module,
                 structure: torch.nn.Module,
                 scheduler=None,
                 dataloader: Iterable[Any]=None,
                 *args, **kwargs) -> None:
        self.epochs = epochs
        self.device = device
        self.models = models
        self.structure = structure
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader

        self._init_structure()
        self._init_device()

    def _init_device(self):
        self.models['teacher']['model'].to(self.device)
        self.models['student']['model'].to(self.device)
        self.distill_structure.to(self.device)
        gaulog.info("Models are loaded to device: " + self.device)

    def _init_structure(self):
        self.distill_structure = self.structure(
            self.models['teacher'],
            self.models['student'],
            self.optimizer,
            self.loss,
        )

    def train(self):
        for data in tqdm(self.dataloader, desc="Training", total=len(self.dataloader)):
            data = self.dataloader.dataset._to_device(data, self.device)
            loss = self.distill_structure.forward(data, mode="train")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
