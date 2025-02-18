import math
import torch

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            max_steps: int,
            warmup_steps: int,
            lr: float,
            batch_size: int,
            last_epoch: int = -1,
            verbose: bool = False,
    ):
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.lr = lr
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        step = self.last_epoch
        max_steps = self.max_steps
        base_lr = self.lr * math.sqrt(self.batch_size / 256)
        if step < self.warmup_steps:
            lr = base_lr * step / self.warmup_steps
        else:
            step -= self.warmup_steps
            max_steps -= self.warmup_steps 
            q = 0.5 * (1+math.cos(math.pi * step / max_steps))
            end_lr = base_lr * 0.001
            lr = base_lr * q + end_lr * (1-q)

        return [lr, lr]