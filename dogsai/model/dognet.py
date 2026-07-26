"""DogBehaviourNet — the behaviour recognition backbone and head.

Shape flow (``small`` preset, 16x160x160 input)::

    input        3 x 16 x 160 x 160
    stem        24 x 16 x  80 x  80    (1,3,3) s2 + (3,1,1) temporal
    stage 1     32 x 16 x  40 x  40    2 blocks, spatial stride 2
    stage 2     64 x  8 x  20 x  20    3 blocks, spatial + temporal stride 2
    stage 3    112 x  8 x  10 x  10    4 blocks
    stage 4    176 x  4 x   5 x   5    3 blocks, temporal stride 2
    head conv  512 x  4 x   5 x   5
    spatial avg 512 x  4                (B, T, C) after transpose
    temporal attention pool -> 512
    classifier -> num_classes

Note the temporal axis downsamples *late* and *slowly* (16 -> 8 -> 4).  Collapsing
time early is the standard way to make a video model fast and simultaneously blind
to the thing it is supposed to detect; keeping four temporal positions all the way
to the pooling layer is what lets the attention head localise a behaviour inside
the clip.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..config import Config, ModelConfig
from .blocks import (
    ConvBNAct3d,
    FactorisedBlock,
    MotionStem,
    TemporalAttentionPool,
    make_divisible,
)


@dataclass(frozen=True)
class StageSpec:
    """One stage of the backbone."""

    channels: int
    blocks: int
    spatial_stride: int
    temporal_stride: int
    expand: float
    spatial_kernel: int = 3
    temporal_kernel: int = 3


# Reference topology at width=1.0, depth=1.0.
DEFAULT_STAGES: tuple[StageSpec, ...] = (
    StageSpec(channels=32, blocks=2, spatial_stride=2, temporal_stride=1, expand=3.0),
    StageSpec(channels=64, blocks=3, spatial_stride=2, temporal_stride=2, expand=4.0),
    StageSpec(channels=112, blocks=4, spatial_stride=2, temporal_stride=1, expand=4.0),
    StageSpec(channels=176, blocks=3, spatial_stride=2, temporal_stride=2, expand=6.0),
)

STEM_CHANNELS = 24


class DogBehaviourNet(nn.Module):
    """Efficient spatio-temporal classifier for dog behaviour.

    Args:
        num_classes: size of the output layer.
        width: channel multiplier.
        depth: block-count multiplier.
        motion_stem: concatenate frame differences to the RGB input.
        temporal_pool: ``attention`` | ``mean`` | ``max``.
    """

    def __init__(
        self,
        num_classes: int,
        width: float = 1.0,
        depth: float = 1.0,
        dropout: float = 0.2,
        drop_path: float = 0.05,
        head_dim: int = 512,
        motion_stem: bool = True,
        temporal_pool: str = "attention",
        attn_heads: int = 4,
        se_ratio: float = 0.25,
        stages: tuple[StageSpec, ...] = DEFAULT_STAGES,
        in_channels: int = 3,
    ):
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        self.num_classes = num_classes
        self.temporal_pool_kind = temporal_pool
        self.motion = MotionStem() if motion_stem else None
        stem_in = in_channels * (2 if motion_stem else 1)

        stem_channels = make_divisible(STEM_CHANNELS * width)
        self.stem = nn.Sequential(
            ConvBNAct3d(stem_in, stem_channels, kernel=(1, 3, 3), stride=(1, 2, 2)),
            # A temporal depthwise conv right at the stem gives every later block
            # access to short-range motion, cheaply (depthwise on 24 channels).
            ConvBNAct3d(stem_channels, stem_channels, kernel=(3, 1, 1), groups=stem_channels),
        )

        total_blocks = sum(max(1, round(s.blocks * depth)) for s in stages)
        block_index = 0
        channels = stem_channels
        backbone: list[nn.Module] = []
        for spec in stages:
            out_channels = make_divisible(spec.channels * width)
            n_blocks = max(1, round(spec.blocks * depth))
            for i in range(n_blocks):
                # Linearly ramp stochastic depth: early layers stay reliable,
                # deep layers get the regularisation.
                dp = drop_path * block_index / max(1, total_blocks - 1)
                backbone.append(
                    FactorisedBlock(
                        in_channels=channels,
                        out_channels=out_channels,
                        spatial_kernel=spec.spatial_kernel,
                        temporal_kernel=spec.temporal_kernel,
                        stride=spec.spatial_stride if i == 0 else 1,
                        expand_ratio=spec.expand,
                        se_ratio=se_ratio,
                        drop_path=dp,
                    )
                )
                channels = out_channels
                block_index += 1
            if spec.temporal_stride > 1:
                # Strided temporal depthwise conv instead of pooling: it keeps a
                # learned low-pass filter in front of the downsample.
                backbone.append(
                    ConvBNAct3d(
                        channels,
                        channels,
                        kernel=(3, 1, 1),
                        stride=(spec.temporal_stride, 1, 1),
                        groups=channels,
                    )
                )
        self.backbone = nn.Sequential(*backbone)

        self.head_conv = ConvBNAct3d(channels, head_dim, (1, 1, 1))
        if temporal_pool == "attention":
            self.temporal_pool: nn.Module = TemporalAttentionPool(
                head_dim, heads=attn_heads, dropout=min(0.1, dropout)
            )
        else:
            self.temporal_pool = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(head_dim, num_classes)
        self.feature_dim = head_dim

        self._init_weights()

    def _init_weights(self) -> None:
        # Deliberately does not touch normalisation weights: BatchNorm already
        # defaults to gamma=1, and FactorisedBlock zero-initialises the gamma of
        # its projection BN to make each residual block start as an identity.
        # Re-setting gamma here would silently undo that.
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    @torch.no_grad()
    def set_prior_bias(self, positive_rate: torch.Tensor | list[float]) -> None:
        """Initialise classifier biases from the training-set positive rate.

        With a zero bias every class starts at p=0.5, so a multi-label head whose
        true positive rate is ~3% spends its first epochs doing nothing but
        pushing 17 outputs down.  Seeding ``b = log(p / (1-p))`` starts the model
        at the correct base rate, which measurably shortens warmup and stops rare
        classes from being crushed before they are ever learned.
        """
        rate = torch.as_tensor(positive_rate, dtype=torch.float32).clamp(1e-4, 1 - 1e-4)
        if rate.numel() != self.num_classes:
            raise ValueError(
                f"expected {self.num_classes} rates, got {rate.numel()}"
            )
        self.classifier.bias.copy_(torch.log(rate / (1 - rate)))

    # -- forward ---------------------------------------------------------
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 3, T, H, W)`` -> ``(B, T', C)`` spatially pooled sequence."""
        if x.ndim != 5:
            raise ValueError(f"expected a 5-D clip (B, C, T, H, W), got {tuple(x.shape)}")
        if self.motion is not None:
            x = self.motion(x)
        x = self.stem(x)
        x = self.backbone(x)
        x = self.head_conv(x)
        x = x.mean(dim=(3, 4))  # spatial global average -> (B, C, T)
        return x.transpose(1, 2)  # -> (B, T, C)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        sequence = self.forward_features(x)
        attention = None
        if isinstance(self.temporal_pool, TemporalAttentionPool):
            if return_attention:
                pooled, attention = self.temporal_pool(sequence, return_weights=True)
            else:
                pooled = self.temporal_pool(sequence)
        elif self.temporal_pool_kind == "max":
            pooled = sequence.max(dim=1).values
        else:
            pooled = sequence.mean(dim=1)
        logits = self.classifier(self.dropout(pooled))
        if return_attention:
            if attention is None:
                attention = sequence.new_full(
                    (sequence.shape[0], sequence.shape[1]), 1.0 / sequence.shape[1]
                )
            return logits, attention
        return logits

    # -- introspection ---------------------------------------------------
    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters()
        return sum(p.numel() for p in params if p.requires_grad or not trainable_only)

    @torch.no_grad()
    def estimate_flops(self, clip_shape: tuple[int, int, int, int]) -> int:
        """Multiply-accumulate count for one clip, by hooking conv/linear layers.

        Reported as MACs (one multiply-add = 1), which is the convention used by
        most papers' "FLOPs" numbers.
        """
        total = 0
        handles = []

        def conv_hook(module, inputs, output):
            nonlocal total
            out_elements = output.numel() / output.shape[0]
            kernel_ops = module.in_channels // module.groups
            for k in module.kernel_size:
                kernel_ops *= k
            total += int(out_elements * kernel_ops)

        def linear_hook(module, inputs, output):
            nonlocal total
            total += int(module.in_features * module.out_features)

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                handles.append(module.register_forward_hook(conv_hook))
            elif isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(linear_hook))

        was_training = self.training
        self.eval()
        device = next(self.parameters()).device
        self(torch.zeros(1, *clip_shape, device=device))
        for handle in handles:
            handle.remove()
        self.train(was_training)
        return total

    def summary(self, clip_shape: tuple[int, int, int, int] | None = None) -> str:
        lines = [
            f"DogBehaviourNet  classes={self.num_classes}  "
            f"feature_dim={self.feature_dim}  motion_stem={self.motion is not None}",
            f"  parameters : {self.num_parameters() / 1e6:.2f} M",
        ]
        if clip_shape is not None:
            macs = self.estimate_flops(clip_shape)
            lines.append(
                f"  MACs/clip  : {macs / 1e9:.2f} G  "
                f"(input {'x'.join(str(s) for s in clip_shape)})"
            )
        return "\n".join(lines)


def build_model(config: Config | ModelConfig, num_classes: int | None = None) -> DogBehaviourNet:
    """Instantiate a model from config."""
    if isinstance(config, Config):
        model_config = config.model
        classes = num_classes if num_classes is not None else config.num_classes
    else:
        model_config = config
        if num_classes is None:
            raise ValueError("num_classes is required when building from a ModelConfig")
        classes = num_classes
    return DogBehaviourNet(
        num_classes=classes,
        width=model_config.width,
        depth=model_config.depth,
        dropout=model_config.dropout,
        drop_path=model_config.drop_path,
        head_dim=model_config.head_dim,
        motion_stem=model_config.motion_stem,
        temporal_pool=model_config.temporal_pool,
        attn_heads=model_config.attn_heads,
        se_ratio=model_config.se_ratio,
    )
