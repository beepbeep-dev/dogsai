"""dogsai — dog behaviour recognition from video.

An efficient spatio-temporal CNN built from scratch in PyTorch, plus the data
plumbing that decides whether it is actually accurate: span-based annotations,
group-aware splitting, a dataset auditor, and sliding-window inference that turns
per-clip scores into a behaviour timeline.

Quick start::

    from dogsai import BehaviourPredictor

    predictor = BehaviourPredictor("runs/dognet/best.pt")
    prediction = predictor.predict("backyard.mp4")
    print(prediction.timeline())
    for span in prediction.spans:
        print(span)

Training from Python::

    from dogsai import ClipDataset, Config, LabelSpace, Trainer, discover_split

    config = Config().apply_preset()
    labels = LabelSpace.from_names(config.behaviours)
    train = ClipDataset(discover_split("data", "train"), labels, config.data, training=True)
    val = ClipDataset(discover_split("data", "val"), labels, config.data)
    Trainer(config, train, val, labels).fit()
"""

from .affect import AffectReading, read_affect
from .audit import AuditReport, audit_annotations, audit_splits
from .cache import CachedClipDataset, CacheSpec, build_cache
from .config import Config, DataConfig, InferenceConfig, ModelConfig, TrainConfig
from .datasets_hub import REGISTRY, download, prepare
from .dataset import (
    Annotation,
    ClipDataset,
    SlidingWindowDataset,
    discover_split,
    load_annotations,
    make_splits,
    save_annotations,
)
from .engine import Trainer, load_checkpoint, save_checkpoint
from .labels import DEFAULT_BEHAVIOURS, BehaviourSpan, LabelSpace
from .metrics import EvalResult, evaluate
from .model import DogBehaviourNet, build_model
from .predict import BehaviourPredictor, VideoPrediction

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # config
    "Config",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "InferenceConfig",
    # labels
    "DEFAULT_BEHAVIOURS",
    "LabelSpace",
    "BehaviourSpan",
    # data
    "Annotation",
    "ClipDataset",
    "SlidingWindowDataset",
    "discover_split",
    "load_annotations",
    "save_annotations",
    "make_splits",
    # clip cache
    "CacheSpec",
    "CachedClipDataset",
    "build_cache",
    # public datasets
    "REGISTRY",
    "download",
    "prepare",
    # audit
    "AuditReport",
    "audit_annotations",
    "audit_splits",
    # affect
    "AffectReading",
    "read_affect",
    # model + training
    "DogBehaviourNet",
    "build_model",
    "Trainer",
    "save_checkpoint",
    "load_checkpoint",
    # eval + inference
    "EvalResult",
    "evaluate",
    "BehaviourPredictor",
    "VideoPrediction",
]
