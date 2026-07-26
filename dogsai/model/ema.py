"""Exponential moving average of model weights.

On small video datasets the validation curve is noisy enough that "best epoch" is
partly luck.  An EMA copy of the weights is a cheap, almost-free variance
reduction — it typically adds a point or two of mAP and makes early stopping mean
something.
"""

from __future__ import annotations

import copy
from typing import Iterator

import torch
import torch.nn as nn


class ModelEMA:
    """Keeps a shadow copy of ``model`` updated as ``ema = d*ema + (1-d)*model``.

    The decay is warmed up over the first steps, because averaging against
    randomly-initialised weights for the first thousand updates just holds the
    EMA back.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: int = 1000):
        self.module = copy.deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)
        self.decay = decay
        self.warmup = max(1, warmup)
        self.updates = 0

    def _current_decay(self) -> float:
        # Ramps 0 -> decay; equivalent to averaging over min(step, 1/(1-decay)).
        return self.decay * (1.0 - torch.exp(torch.tensor(-self.updates / self.warmup)).item())

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self._current_decay()
        ema_state = self.module.state_dict()
        for key, value in model.state_dict().items():
            shadow = ema_state[key]
            if not shadow.dtype.is_floating_point:
                shadow.copy_(value)  # integer buffers (num_batches_tracked)
                continue
            shadow.mul_(decay).add_(value.detach(), alpha=1.0 - decay)

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state)

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.module.parameters()

    def to(self, *args, **kwargs) -> "ModelEMA":
        self.module.to(*args, **kwargs)
        return self
