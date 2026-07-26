"""Model definitions for dogsai."""

from .blocks import (
    ConvBNAct3d,
    DropPath,
    FactorisedBlock,
    MotionStem,
    SqueezeExcite3d,
    TemporalAttentionPool,
    make_divisible,
)
from .dognet import DEFAULT_STAGES, DogBehaviourNet, StageSpec, build_model
from .ema import ModelEMA

__all__ = [
    "ConvBNAct3d",
    "DropPath",
    "FactorisedBlock",
    "MotionStem",
    "SqueezeExcite3d",
    "TemporalAttentionPool",
    "make_divisible",
    "DEFAULT_STAGES",
    "DogBehaviourNet",
    "StageSpec",
    "build_model",
    "ModelEMA",
]
