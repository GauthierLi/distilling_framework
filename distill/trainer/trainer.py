import os
import torch 

from tqdm import tqdm 
from distill.loger import gaulog
from distill.utils import make_save_dir
from typing import Dict, Any, Iterable, List, Tuple

class BaseTrainer(object):
    """
    
        Base class for training and validating the model.
    ---
    Parameters:
        - device (str): Device to use for training (e.g., "cuda" or "cpu").
        - epochs (int): Number of training epochs.
        - models (Dict[str, Dict[str, Any]]): Dictionary containing the teacher and student models.
        - optimizer (torch.optim.Optimizer): Optimizer for training.
        - loss (torch.nn.Module): Loss function for training.
        - structure (torch.nn.Module): Distillation structure.
        - scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        - dataloader (Iterable[Any]): DataLoader for training data.
        - validate_period (int): Period for validation. Validation after every `validate_period` epochs.
        - max_saved (int): Maximum number of saved checkpoints. '-1' for every epoch.
        - save_dir (str): Directory to save checkpoints and logs.
    """
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
                 max_saved: int=-1,
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

        self.max_saved = max_saved
        self.saved_epochs = []

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
        gaulog.info("-- Start training --")
        for epoch in range(self.epochs):
            gaulog.info(f"-- Epoch {epoch + 1}/{self.epochs} training --")
            self.train_one_epochs()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.validate_period > 0 and (epoch + 1) % self.validate_period == 0:
                gaulog.info("Validating...")
                self.validate_one_epochs()
            gaulog.info(f"Epoch {epoch + 1} finished.")
            self.save_ckpts(epoch + 1)

        gaulog.info("Training finished.")

    def save_ckpts(self, epoch: int):
        # save all checkpoints
        ckpt_all_save_path = os.path.join(
            self.save_dir, "ckpt", f"all_epoch_{epoch}.pth"
        )
        ckpt_student_save_path = os.path.join(
            self.save_dir, "ckpt", f"student_epoch_{epoch}.pth"
        )

        all_params = {
            "teacher": self.models["teacher"]["model"].state_dict(),
            "student": self.models["student"]["model"].state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }

        gaulog.info("-- Saving checkpoints --")
        torch.save(all_params, ckpt_all_save_path)
        gaulog.info(f"Checkpoint saved to {ckpt_all_save_path}")
        torch.save(self.models["student"]["model"].state_dict(), ckpt_student_save_path)
        gaulog.info(f"Student Checkpoint saved to {ckpt_student_save_path}")

        self.saved_epochs.append(epoch)
        if len(self.saved_epochs) > self.max_saved and self.max_saved != -1:
            oldest_epoch = self.saved_epochs.pop(0)
            oldest_ckpt_path = os.path.join(
                self.save_dir, "ckpt", f"all_epoch_{oldest_epoch}.pth"
            )
            if os.path.exists(oldest_ckpt_path):
                os.remove(oldest_ckpt_path)
                gaulog.info(f"Removed oldest checkpoint: {oldest_ckpt_path}")
