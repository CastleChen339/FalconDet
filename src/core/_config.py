"""Runtime configuration interfaces used by FalconDet."""

from pathlib import Path

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


__all__ = ["BaseConfig"]


class BaseConfig:
    def __init__(self) -> None:
        self.task: str | None = None

        self._model: nn.Module | None = None
        self._postprocessor: nn.Module | None = None
        self._criterion: nn.Module | None = None
        self._optimizer: Optimizer | None = None
        self._lr_scheduler: LRScheduler | None = None
        self._lr_warmup_scheduler = None
        self._train_dataloader: DataLoader | None = None
        self._val_dataloader: DataLoader | None = None
        self._ema: nn.Module | None = None
        self._writer: SummaryWriter | None = None

        self.resume: str | None = None
        self.tuning: str | None = None

        self.epochs: int | None = None
        self.last_epoch: int = -1
        self.freeze_backbone_epochs: int | None = None
        self.freeze_encoder_epochs: int = 0
        self.max_eval_batches: int | None = None

        self.use_ema: bool = False
        self.ema_decay: float = 0.9999
        self.ema_warmups: int = 2000
        self.sync_bn: bool = False
        self.clip_max_norm: float = 0.0
        self.find_unused_parameters: bool | None = None

        self.seed: int | None = None
        self.print_freq: int | None = None
        self.checkpoint_freq: int = 1
        self.eval_interval: int = 1
        self.best_metric: str = "point_f1"
        self.best_metrics: list[str] | None = None
        self.best_mode: str = "max"
        self.log_branch_grad_norms: bool = False
        self.branch_grad_norm_interval: int = 50
        self.output_dir: str | None = None
        self.summary_dir: str | None = None
        self.device: str = ""

    @property
    def model(self) -> nn.Module | None:
        return self._model

    @model.setter
    def model(self, module: nn.Module) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError(f"{type(module)} must be an nn.Module")
        self._model = module

    @property
    def postprocessor(self) -> nn.Module | None:
        return self._postprocessor

    @postprocessor.setter
    def postprocessor(self, module: nn.Module) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError(f"{type(module)} must be an nn.Module")
        self._postprocessor = module

    @property
    def criterion(self) -> nn.Module | None:
        return self._criterion

    @criterion.setter
    def criterion(self, module: nn.Module) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError(f"{type(module)} must be an nn.Module")
        self._criterion = module

    @property
    def optimizer(self) -> Optimizer | None:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer)} must be an Optimizer")
        self._optimizer = optimizer

    @property
    def lr_scheduler(self) -> LRScheduler | None:
        return self._lr_scheduler

    @lr_scheduler.setter
    def lr_scheduler(self, scheduler: LRScheduler) -> None:
        if not isinstance(scheduler, LRScheduler):
            raise TypeError(f"{type(scheduler)} must be an LRScheduler")
        self._lr_scheduler = scheduler

    @property
    def lr_warmup_scheduler(self):
        return self._lr_warmup_scheduler

    @lr_warmup_scheduler.setter
    def lr_warmup_scheduler(self, scheduler) -> None:
        self._lr_warmup_scheduler = scheduler

    @property
    def train_dataloader(self) -> DataLoader | None:
        return self._train_dataloader

    @train_dataloader.setter
    def train_dataloader(self, loader: DataLoader) -> None:
        self._train_dataloader = loader

    @property
    def val_dataloader(self) -> DataLoader | None:
        return self._val_dataloader

    @val_dataloader.setter
    def val_dataloader(self, loader: DataLoader) -> None:
        self._val_dataloader = loader

    @property
    def ema(self) -> nn.Module | None:
        if self._ema is None and self.use_ema and self.model is not None:
            from ..optim import ModelEMA

            self._ema = ModelEMA(self.model, self.ema_decay, self.ema_warmups)
        return self._ema

    @ema.setter
    def ema(self, module: nn.Module) -> None:
        self._ema = module

    @property
    def writer(self) -> SummaryWriter | None:
        if self._writer is None:
            if self.summary_dir:
                self._writer = SummaryWriter(self.summary_dir)
            elif self.output_dir:
                self._writer = SummaryWriter(Path(self.output_dir) / "summary")
        return self._writer

    @writer.setter
    def writer(self, writer: SummaryWriter) -> None:
        if not isinstance(writer, SummaryWriter):
            raise TypeError(f"{type(writer)} must be a SummaryWriter")
        self._writer = writer

    def __repr__(self) -> str:
        return "".join(
            f"{key}: {value}\n"
            for key, value in self.__dict__.items()
            if not key.startswith("_")
        )
