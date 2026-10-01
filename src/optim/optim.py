import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from ..core import register


__all__ = ["AdamW", "MultiStepLR"]


AdamW = register()(optim.AdamW)
MultiStepLR = register()(lr_scheduler.MultiStepLR)
