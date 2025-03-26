import torch 

from typing import Dict, Any
from distill.utils import gaulog

class BaseTrainer(object):
    def __init__(self,
                 device: str,
                 epochs: int,
                 models: Dict[str, Dict[str, Any]],
                 optimizer: torch.optim.Optimizer,
                 loss: torch.nn.Module,
                 structure: torch.nn.Module,
                 scheduler=None,
                 ):
        self.epochs = epochs
        self.device = device
        self.models = models
        self.structure = structure
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.distill_structure = None

        self._init_structure()
        self._init_device()

    def _init_device(self):
        self.models['teacher']['model'].to(self.device)
        self.models['student']['model'].to(self.device)
        self.distill_structure.to(self.device)
        gaulog.info("Models are loaded to device: ", self.device)

    def _init_structure(self):
        self.distill_structure = self.structure(
            self.models['teacher'],
            self.models['student'],
            self.loss,
            self.optimizer,
        )

    def train(self):
        ...
