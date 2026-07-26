"""Building blocks for DogBehaviourNet.

The design target is behaviour recognition on CPU-or-modest-GPU budgets, which
rules out plain 3-D convolutions: a 3x3x3 kernel costs 3x a 2-D one for the same
channel count, and most of that cost buys very little.  Two factorisations do the
heavy lifting here:

* **Spatial/temporal separation** — a ``(1,k,k)`` depthwise convolution followed
  by a ``(kt,1,1)`` depthwise convolution spans the same receptive field as
  ``(kt,k,k)`` at ``(k*k + kt)/(k*k*kt)`` of the parameters, and the extra
  non-linearity between them is a documented win (the R(2+1)D result).
* **Depthwise/pointwise separation** — inverted residual bottlenecks, so the
  expensive channel mixing happens at 1x1x1.

Everything is written directly against ``torch.nn``; no torchvision, no timm, no
pretrained weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(value: float, divisor: int = 8, min_value: int | None = None) -> int:
    """Round a scaled channel count to a multiple of ``divisor``.

    Width multipliers produce ugly numbers like 43.2 channels; kernels are much
    happier with multiples of 8, and never dropping below 90% avoids silently
    halving a layer.
    """
    if min_value is None:
        min_value = divisor
    out = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if out < 0.9 * value:
        out += divisor
    return int(out)


class ConvBNAct3d(nn.Sequential):
    """Conv3d -> BatchNorm3d -> activation, with padding worked out for you."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: tuple[int, int, int] = (1, 1, 1),
        stride: tuple[int, int, int] = (1, 1, 1),
        groups: int = 1,
        activation: type[nn.Module] | None = nn.Hardswish,
        bn_weight_init: float = 1.0,
    ):
        padding = tuple(k // 2 for k in kernel)
        layers: list[nn.Module] = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
        ]
        nn.init.constant_(layers[1].weight, bn_weight_init)
        if activation is not None:
            layers.append(activation(inplace=True) if _takes_inplace(activation) else activation())
        super().__init__(*layers)


def _takes_inplace(activation: type[nn.Module]) -> bool:
    return activation in (nn.ReLU, nn.ReLU6, nn.Hardswish, nn.SiLU, nn.Hardsigmoid)


class SqueezeExcite3d(nn.Module):
    """Channel gating from a global spatio-temporal average.

    For behaviour this earns its keep: the gate is computed over time as well as
    space, so it can suppress channels that are only ever active in still frames.
    """

    def __init__(self, channels: int, ratio: float = 0.25):
        super().__init__()
        hidden = make_divisible(channels * ratio, 8, min_value=8)
        self.reduce = nn.Conv3d(channels, hidden, 1)
        self.expand = nn.Conv3d(hidden, channels, 1)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Hardsigmoid(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=(2, 3, 4), keepdim=True)
        scale = self.gate(self.expand(self.act(self.reduce(scale))))
        return x * scale


class DropPath(nn.Module):
    """Stochastic depth: drop whole residual branches per sample.

    Preferred over heavier dropout inside a video backbone — clips within a batch
    are highly correlated, and dropping the branch keeps BatchNorm statistics
    cleaner than dropping activations.
    """

    def __init__(self, probability: float = 0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability <= 0.0:
            return x
        keep = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep

    def extra_repr(self) -> str:
        return f"p={self.probability:.3f}"


class FactorisedBlock(nn.Module):
    """Inverted residual with separated spatial and temporal depthwise convs.

    ``expand(1x1x1) -> DW(1xkxk) -> DW(ktx1x1) -> SE -> project(1x1x1)``

    The temporal depthwise conv is skipped when ``temporal_kernel == 1``, which is
    how the nano preset trades some motion modelling for speed.  The projection's
    BatchNorm is zero-initialised so each block starts as an identity — a
    from-scratch model with 12+ blocks trains noticeably more stably that way.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_kernel: int = 3,
        temporal_kernel: int = 3,
        stride: int = 1,
        expand_ratio: float = 4.0,
        se_ratio: float = 0.25,
        drop_path: float = 0.0,
    ):
        super().__init__()
        hidden = make_divisible(in_channels * expand_ratio)
        self.use_residual = stride == 1 and in_channels == out_channels

        self.expand = (
            ConvBNAct3d(in_channels, hidden, (1, 1, 1))
            if hidden != in_channels
            else nn.Identity()
        )
        self.spatial = ConvBNAct3d(
            hidden,
            hidden,
            kernel=(1, spatial_kernel, spatial_kernel),
            stride=(1, stride, stride),
            groups=hidden,
        )
        self.temporal = (
            ConvBNAct3d(
                hidden,
                hidden,
                kernel=(temporal_kernel, 1, 1),
                groups=hidden,
            )
            if temporal_kernel > 1
            else nn.Identity()
        )
        self.se = SqueezeExcite3d(hidden, se_ratio) if se_ratio > 0 else nn.Identity()
        self.project = ConvBNAct3d(
            hidden,
            out_channels,
            (1, 1, 1),
            activation=None,
            bn_weight_init=0.0 if self.use_residual else 1.0,
        )
        self.drop_path = DropPath(drop_path) if self.use_residual else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.spatial(out)
        out = self.temporal(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            return x + self.drop_path(out)
        return out


class MotionStem(nn.Module):
    """Concatenate RGB with successive frame differences.

    Behaviours like tail-wagging, shaking-off and digging are defined by small
    fast motion that a low-resolution appearance stream barely registers.  An
    explicit difference channel hands that signal to the first layer for the price
    of one subtraction, with no extra decoding and no second backbone — the cheap
    90% of a two-stream network.
    """

    def __init__(self, gain: float = 2.0):
        super().__init__()
        # Differences are small; a learnable gain lets the network scale them into
        # the same range as the normalised RGB it is concatenated with.
        self.gain = nn.Parameter(torch.tensor(float(gain)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = x[:, :, 1:] - x[:, :, :-1]
        diff = F.pad(diff, (0, 0, 0, 0, 1, 0))  # repeat-pad the first timestep
        return torch.cat([x, diff * self.gain], dim=1)


class TemporalAttentionPool(nn.Module):
    """Attention pooling over time with a learned query.

    A clip is 16 frames of which maybe 4 contain the jump.  Mean pooling dilutes
    those frames by 4x; max pooling keeps one and throws away context.  A learned
    query attending over the temporal sequence lets the model decide which frames
    carry the behaviour, and the returned weights double as a free
    interpretability signal (see :meth:`forward` returning ``weights``).
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if dim % heads != 0:
            heads = 1
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.query = nn.Parameter(torch.randn(heads, self.head_dim) * 0.02)
        self.keys = nn.Linear(dim, dim, bias=False)
        self.values = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, return_weights: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, C)
        b, t, _ = x.shape
        x = self.norm(x)
        k = self.keys(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        v = self.values(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.einsum("hd,bhtd->bht", self.query, k) * self.scale
        weights = scores.softmax(dim=-1)
        pooled = torch.einsum("bht,bhtd->bhd", self.dropout(weights), v)
        pooled = self.out(pooled.reshape(b, -1))
        if return_weights:
            return pooled, weights.mean(dim=1)
        return pooled
