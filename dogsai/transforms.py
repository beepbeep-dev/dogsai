"""Clip-consistent augmentation.

The one rule that matters for video: **a geometric or photometric transform must
be sampled once per clip, not once per frame.**  Jittering each frame
independently injects fake motion, and a motion model will happily learn that
noise instead of the behaviour.  Every op here takes its parameters once and
applies them to the whole clip.

Input to the pipeline is ``uint8 (T, H, W, 3)`` RGB; output is a normalised
float tensor ``(3, T, H, W)`` — channels-first with time as the depth axis, which
is what the 3-D convolutions in :mod:`dogsai.model` expect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# ImageNet statistics.  We train from scratch, but these are still a sane
# whitening of natural-image pixel statistics.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

_GREY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _resize_clip(clip: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize every frame of a clip to ``(height, width)``."""
    if clip.shape[1] == height and clip.shape[2] == width:
        return clip
    try:
        import cv2

        # Area for downscale (anti-aliased), linear for upscale.
        shrinking = width < clip.shape[2] or height < clip.shape[1]
        interp = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        return np.stack([cv2.resize(f, (width, height), interpolation=interp) for f in clip])
    except ImportError:
        return _resize_clip_numpy(clip, width, height)


def _resize_clip_numpy(clip: np.ndarray, width: int, height: int) -> np.ndarray:
    """Nearest-neighbour fallback so the package works without OpenCV."""
    ys = (np.linspace(0, clip.shape[1] - 1, height)).round().astype(np.int64)
    xs = (np.linspace(0, clip.shape[2] - 1, width)).round().astype(np.int64)
    return clip[:, ys][:, :, xs]


def random_resized_crop(
    clip: np.ndarray,
    size: int,
    scale: tuple[float, float] = (0.65, 1.0),
    ratio: tuple[float, float] = (3 / 4, 4 / 3),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Crop one random box out of the clip, then resize to ``size``."""
    rng = rng or np.random.default_rng()
    _, h, w, _ = clip.shape
    area = float(h * w)
    for _ in range(10):
        target = area * float(rng.uniform(*scale))
        log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
        aspect = float(np.exp(rng.uniform(*log_ratio)))
        cw = int(round(np.sqrt(target * aspect)))
        ch = int(round(np.sqrt(target / aspect)))
        if 0 < cw <= w and 0 < ch <= h:
            x0 = int(rng.integers(0, w - cw + 1))
            y0 = int(rng.integers(0, h - ch + 1))
            cropped = clip[:, y0 : y0 + ch, x0 : x0 + cw]
            return _resize_clip(cropped, size, size)
    return center_crop(clip, size)


def center_crop(clip: np.ndarray, size: int) -> np.ndarray:
    """Resize the short side to ``size`` then take the centre square."""
    _, h, w, _ = clip.shape
    if h == 0 or w == 0:
        raise ValueError("empty clip")
    scale = size / min(h, w)
    new_w, new_h = max(size, int(round(w * scale))), max(size, int(round(h * scale)))
    resized = _resize_clip(clip, new_w, new_h)
    y0 = (new_h - size) // 2
    x0 = (new_w - size) // 2
    return resized[:, y0 : y0 + size, x0 : x0 + size]


def hflip(clip: np.ndarray) -> np.ndarray:
    return clip[:, :, ::-1]


def colour_jitter(
    clip: np.ndarray, strength: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Brightness / contrast / saturation jitter, one setting for the whole clip."""
    if strength <= 0:
        return clip
    rng = rng or np.random.default_rng()
    out = clip.astype(np.float32)
    brightness = 1.0 + float(rng.uniform(-strength, strength))
    contrast = 1.0 + float(rng.uniform(-strength, strength))
    saturation = 1.0 + float(rng.uniform(-strength, strength))

    out *= brightness
    mean = out.mean()
    out = (out - mean) * contrast + mean
    grey = out @ _GREY_WEIGHTS
    out = grey[..., None] + (out - grey[..., None]) * saturation
    return np.clip(out, 0, 255).astype(np.uint8)


def to_greyscale(clip: np.ndarray) -> np.ndarray:
    grey = (clip.astype(np.float32) @ _GREY_WEIGHTS)[..., None]
    return np.repeat(np.clip(grey, 0, 255).astype(np.uint8), 3, axis=-1)


def to_tensor(clip: np.ndarray) -> torch.Tensor:
    """``uint8 (T, H, W, 3)`` -> normalised float ``(3, T, H, W)``."""
    array = np.ascontiguousarray(clip)
    tensor = torch.from_numpy(array).float().div_(255.0)
    tensor = tensor.permute(3, 0, 1, 2)  # T,H,W,C -> C,T,H,W
    mean = torch.tensor(MEAN, dtype=tensor.dtype).view(3, 1, 1, 1)
    std = torch.tensor(STD, dtype=tensor.dtype).view(3, 1, 1, 1)
    return tensor.sub_(mean).div_(std)


def denormalise(tensor: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`to_tensor` normalisation, for visual debugging."""
    mean = torch.tensor(MEAN, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1, 1)
    std = torch.tensor(STD, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1, 1)
    return (tensor * std + mean).clamp_(0, 1)


def random_erase(
    tensor: torch.Tensor,
    probability: float,
    rng: np.random.Generator | None = None,
    scale: tuple[float, float] = (0.02, 0.15),
) -> torch.Tensor:
    """Erase one spatial box across every frame ("tube" erasing).

    Erasing a different box per frame would look like a flickering occluder;
    erasing the same box for the whole clip looks like a real obstruction, which
    is what we want the model to be robust to.
    """
    if probability <= 0:
        return tensor
    rng = rng or np.random.default_rng()
    if rng.random() > probability:
        return tensor
    _, _, h, w = tensor.shape
    area = h * w * float(rng.uniform(*scale))
    aspect = float(np.exp(rng.uniform(np.log(0.4), np.log(2.5))))
    eh = min(h, max(1, int(round(np.sqrt(area * aspect)))))
    ew = min(w, max(1, int(round(np.sqrt(area / aspect)))))
    y0 = int(rng.integers(0, h - eh + 1))
    x0 = int(rng.integers(0, w - ew + 1))
    tensor[:, :, y0 : y0 + eh, x0 : x0 + ew] = torch.randn(
        tensor.shape[0], tensor.shape[1], eh, ew, dtype=tensor.dtype
    )
    return tensor


@dataclass
class ClipTransform:
    """The train/eval augmentation pipeline as a single callable."""

    size: int = 160
    training: bool = False
    scale: tuple[float, float] = (0.65, 1.0)
    hflip_p: float = 0.5
    jitter: float = 0.25
    grey_p: float = 0.05
    erase_p: float = 0.15

    def __call__(
        self, clip: np.ndarray, rng: np.random.Generator | None = None
    ) -> torch.Tensor:
        rng = rng or np.random.default_rng()
        if not self.training:
            return to_tensor(center_crop(clip, self.size))

        clip = random_resized_crop(clip, self.size, self.scale, rng=rng)
        if self.hflip_p and rng.random() < self.hflip_p:
            clip = hflip(clip)
        if self.grey_p and rng.random() < self.grey_p:
            clip = to_greyscale(clip)
        elif self.jitter:
            clip = colour_jitter(clip, self.jitter, rng=rng)
        tensor = to_tensor(clip)
        return random_erase(tensor, self.erase_p, rng=rng)


def mixup_batch(
    clips: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convex-combine clips and their (already one-hot/multi-hot) targets.

    Mixing whole clips rather than frames keeps each mixed sample temporally
    coherent — two overlaid videos, not a shuffled mess.
    """
    if alpha <= 0 or clips.shape[0] < 2:
        return clips, targets
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)  # keep the dominant sample dominant
    perm = torch.randperm(clips.shape[0], generator=generator, device=clips.device)
    mixed = clips.mul(lam).add_(clips[perm], alpha=1.0 - lam)
    mixed_targets = targets.float().mul(lam).add_(targets[perm].float(), alpha=1.0 - lam)
    return mixed, mixed_targets
