import torch 

from tqdm import tqdm 
from distill.loger import gaulog
from distill.utils import make_save_dir
from typing import Dict, Any, Iterable, List, Tuple

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
                 validate_period: int=0,
                 save_dir: str="workdirs/output",
                 *args, **kwargs) -> None:
        self.epochs = epochs
        self.device = device
        self.models = models
        self.structure = structure
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.save_dir = save_dir
        self.validate_period = validate_period
        make_save_dir(save_dir)

        self._init_structure()
        self._init_device()

    def _init_device(self):
        self.models['teacher']['model'].to(self.device)
        self.models['student']['model'].to(self.device)
        for head in \
            self.models['teacher']["teacher_out_layers"]["heads"]:
            head.to(self.device)

        for head in \
            self.models['student']["student_out_layers"]["heads"]:
            head.to(self.device)
        self.distill_structure.to(self.device)
        gaulog.info("Models are loaded to device: " + self.device)

    def _init_structure(self):
        self.distill_structure = self.structure(
            self.models['teacher'],
            self.models['student'],
            self.loss,
        )

    def train_one_epochs(self):
        for data in tqdm(self.dataloader, desc="Training", total=len(self.dataloader)):
            data = self.dataloader.dataset._to_device(data, self.device)
            loss = self.distill_structure.forward(data, mode="train")
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def validate_one_epochs(self):
        ...

    def train(self):
        gaulog.info("Start training...")
        for epoch in range(self.epochs):
            gaulog.info(f"Epoch {epoch + 1}/{self.epochs}")
            self.train_one_epochs()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.validate_period > 0 and (epoch + 1) % self.validate_period == 0:
                gaulog.info("Validating...")
                self.validate_one_epochs()
            gaulog.info(f"Epoch {epoch + 1} finished.")
        gaulog.info("Training finished.")
