"""Configuration objects.

Everything the pipeline needs is described by plain dataclasses so a run is
reproducible from a single JSON/YAML file, and so a checkpoint can carry its own
config around (see :func:`Config.from_dict`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, get_origin, get_type_hints

from .labels import DEFAULT_BEHAVIOURS

Task = Literal["multiclass", "multilabel"]


@dataclass
class DataConfig:
    root: str = "data"
    """Dataset root.  Either ``root/<split>/<behaviour>/clip.mp4`` (folder mode)
    or ``root/<split>.jsonl`` with one annotation record per line."""

    train_split: str = "train"
    val_split: str = "val"

    clip_frames: int = 16
    """Frames fed to the network per clip."""

    frame_stride: int = 2
    """Sample every N-th source frame, so a 16-frame clip spans 32 source
    frames (~1.1s at 30fps).  Long enough for a gait cycle, short enough to stay
    cheap."""

    image_size: int = 160
    train_scale: tuple[float, float] = (0.65, 1.0)
    """Random-resized-crop area range, applied identically to every frame of a
    clip so motion is not destroyed."""

    sampling: Literal["tsn", "contiguous"] = "tsn"
    """``tsn`` splits the clip window into N segments and jitters one frame per
    segment (better coverage of slow behaviours); ``contiguous`` takes a dense
    run of frames (better for fast, fine motion)."""

    hflip: float = 0.5
    colour_jitter: float = 0.25
    grey: float = 0.05
    erase: float = 0.15
    temporal_reverse: float = 0.0
    """Time-reversal augmentation.  Off by default: it turns ``lying_down ->
    standing`` into ``standing -> lying_down``, which flips the meaning of some
    behaviours."""

    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    cache_index: bool = True
    """Cache per-file frame counts next to the dataset so startup is O(1) after
    the first run."""


@dataclass
class ModelConfig:
    preset: Literal["nano", "small", "base", "custom"] = "small"
    width: float = 1.0
    depth: float = 1.0
    dropout: float = 0.2
    drop_path: float = 0.05
    head_dim: int = 512
    motion_stem: bool = True
    """Feed frame differences alongside RGB.  Costs one subtraction and buys a
    lot on motion-defined behaviours (wagging, shaking, digging)."""

    temporal_pool: Literal["attention", "mean", "max"] = "attention"
    attn_heads: int = 4
    se_ratio: float = 0.25


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 8
    accum_steps: int = 1
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    warmup_epochs: float = 2.0
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    mixup: float = 0.2
    """Clip-level mixup alpha (0 disables)."""
    focal_gamma: float = 1.5
    """Only used for ``multilabel``; damps the easy-negative flood."""
    class_balanced: bool = True
    ema_decay: float = 0.999
    amp: bool = True
    channels_last: bool = True
    early_stop_patience: int = 12
    seed: int = 1337
    out_dir: str = "runs/dognet"
    log_every: int = 10
    compile: bool = False
    """torch.compile the model.  Big win on GPU, usually a loss on CPU."""


@dataclass
class InferenceConfig:
    window_stride: float = 0.5
    """Sliding-window hop as a fraction of the clip span.  0.5 = 50% overlap."""
    batch_size: int = 8
    smooth: int = 3
    """Median-filter width (in windows) over the score timeline."""
    threshold: float = 0.5
    min_duration: float = 0.35
    """Drop spans shorter than this; kills single-window flicker."""
    merge_gap: float = 0.25
    """Bridge same-behaviour spans separated by less than this."""
    tta_hflip: bool = False


@dataclass
class Config:
    task: Task = "multilabel"
    behaviours: list[str] = field(default_factory=lambda: list(DEFAULT_BEHAVIOURS))
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferenceConfig = field(default_factory=InferenceConfig)

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        return _build(cls, raw)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        text = Path(path).read_text()
        if str(path).endswith((".yaml", ".yml")):
            import yaml  # optional dependency

            raw = yaml.safe_load(text)
        else:
            raw = json.loads(text)
        return cls.from_dict(raw or {})

    # -- derived ---------------------------------------------------------
    @property
    def num_classes(self) -> int:
        return len(self.behaviours)

    @property
    def clip_span_frames(self) -> int:
        """How many *source* frames one clip covers."""
        return self.data.clip_frames * self.data.frame_stride

    def apply_preset(self) -> "Config":
        """Resolve ``model.preset`` into concrete width/depth/resolution."""
        presets = {
            # preset: (width, depth, image_size, clip_frames, head_dim)
            "nano": (0.5, 0.75, 128, 12, 320),
            "small": (0.75, 1.0, 160, 16, 512),
            "base": (1.0, 1.25, 192, 24, 768),
        }
        if self.model.preset in presets:
            w, d, size, frames, head = presets[self.model.preset]
            self.model.width = w
            self.model.depth = d
            self.data.image_size = size
            self.data.clip_frames = frames
            self.model.head_dim = head
        return self


@lru_cache(maxsize=None)
def _hints(tp: type) -> dict[str, Any]:
    """Resolved type hints for a dataclass.

    ``from __future__ import annotations`` turns every annotation into a string,
    so ``dataclasses.Field.type`` is ``"DataConfig"`` rather than the class.
    ``get_type_hints`` evaluates them against the defining module, which is what
    makes the nested rebuild below actually recurse.
    """
    return get_type_hints(tp)


def _build(tp: type, raw: Any) -> Any:
    """Recursively instantiate nested dataclasses from plain dicts."""
    if not is_dataclass(tp) or not isinstance(raw, dict):
        return raw
    known = {f.name for f in fields(tp)}
    hints = _hints(tp)
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in known:
            raise KeyError(f"unknown config key {key!r} for {tp.__name__}")
        hint = hints.get(key)
        if is_dataclass(hint) and isinstance(value, dict):
            kwargs[key] = _build(hint, value)
        elif get_origin(hint) is tuple and isinstance(value, list):
            # JSON has no tuples; restore them so equality and unpacking behave.
            kwargs[key] = tuple(value)
        else:
            kwargs[key] = value
    return tp(**kwargs)
