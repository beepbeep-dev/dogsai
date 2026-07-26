"""Datasets, annotation loading and samplers.

Two annotation modes, in increasing order of how accurate a model they produce:

**Folder mode** — ``root/train/running/clip0.mp4``.  Zero-effort, and fine for a
first pass, but it forces a whole-clip label: if the dog runs for 2s of a 10s
clip, 8s of that clip are mislabelled, and the model learns the background.

**Span mode** (recommended) — ``root/train.jsonl``, one record per labelled
interval::

    {"video": "yard_cam_03.mp4", "start": 12.5, "end": 18.0,
     "labels": ["running", "barking"], "group": "yard_cam_03"}

Spans are what make the labels *accurate*: the clip sampler only ever draws
frames from inside a labelled interval, so every training frame actually shows
the behaviour.  Span mode also supports explicit negatives (``"labels": []``),
which teach the model what "no listed behaviour" looks like — without them a
detector fires constantly on unlabelled footage.

The ``group`` field is the other accuracy lever, and the one most easily
forgotten: two clips cut from the same source video are near-duplicates.  Put one
in train and one in val and the val score is inflated by leakage.  Grouping is
respected by :func:`make_splits` and audited by :mod:`dogsai.audit`.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .config import Config, DataConfig
from .labels import LabelSpace
from .transforms import ClipTransform
from .video import MetaCache, VideoMeta, find_videos, sample_indices, window_starts


@dataclass
class Annotation:
    """One labelled interval of one video."""

    video: str
    labels: list[str]
    start: float = 0.0
    end: float | None = None
    group: str | None = None
    subject: str | None = None
    weight: float = 1.0
    meta: dict = field(default_factory=dict)

    @property
    def group_key(self) -> str:
        """Leakage-safe grouping key: explicit group, else the source filename."""
        return self.group or Path(self.video).stem

    def to_dict(self) -> dict:
        out = {
            "video": self.video,
            "labels": list(self.labels),
            "start": round(self.start, 4),
        }
        if self.end is not None:
            out["end"] = round(self.end, 4)
        if self.group:
            out["group"] = self.group
        if self.subject:
            out["subject"] = self.subject
        if self.weight != 1.0:
            out["weight"] = self.weight
        if self.meta:
            out["meta"] = self.meta
        return out

    @classmethod
    def from_dict(cls, raw: dict, root: Path | None = None) -> "Annotation":
        if "video" not in raw:
            raise ValueError(f"annotation record missing 'video': {raw}")
        labels = raw.get("labels", raw.get("label", []))
        if isinstance(labels, str):
            labels = [labels]
        video = str(raw["video"])
        if root is not None and not Path(video).is_absolute():
            video = str(root / video)
        return cls(
            video=video,
            labels=[str(x) for x in labels],
            start=float(raw.get("start", 0.0) or 0.0),
            end=None if raw.get("end") in (None, "") else float(raw["end"]),
            group=raw.get("group"),
            subject=raw.get("subject"),
            weight=float(raw.get("weight", 1.0)),
            meta=raw.get("meta", {}) or {},
        )


def load_annotations(path: str | Path, root: Path | None = None) -> list[Annotation]:
    """Read a ``.jsonl`` (one record per line) or ``.json`` (list) annotation file."""
    path = Path(path)
    root = root if root is not None else path.parent
    text = path.read_text().strip()
    if not text:
        return []
    records: list[dict]
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        blob = json.loads(text)
        records = blob["annotations"] if isinstance(blob, dict) else blob
    return [Annotation.from_dict(r, root) for r in records]


def save_annotations(path: str | Path, annotations: Sequence[Annotation]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for ann in annotations:
            handle.write(json.dumps(ann.to_dict()) + "\n")
    return path


def annotations_from_folders(root: str | Path) -> list[Annotation]:
    """Build whole-clip annotations from ``root/<behaviour>/<clip>.mp4``."""
    root = Path(root)
    out: list[Annotation] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for video in find_videos(class_dir):
            out.append(
                Annotation(
                    video=str(video),
                    labels=[class_dir.name],
                    group=video.stem,
                )
            )
    return out


def discover_split(root: str | Path, split: str) -> list[Annotation]:
    """Find annotations for a split, preferring span mode over folder mode."""
    root = Path(root)
    for suffix in (".jsonl", ".json"):
        candidate = root / f"{split}{suffix}"
        if candidate.exists():
            return load_annotations(candidate, root)
    folder = root / split
    if folder.is_dir():
        return annotations_from_folders(folder)
    raise FileNotFoundError(
        f"no annotations for split {split!r} under {root} "
        f"(looked for {split}.jsonl, {split}.json and {split}/)"
    )


# ---------------------------------------------------------------------------
# samples
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    """A drawable training example: a frame range plus a target vector."""

    annotation: Annotation
    meta: VideoMeta
    first_frame: int
    last_frame: int
    target: np.ndarray

    @property
    def n_available(self) -> int:
        return self.last_frame - self.first_frame + 1


class ClipDataset(Dataset):
    """Samples fixed-length clips from annotated videos.

    Each ``__getitem__`` re-samples the frame window inside the annotated span,
    so repeated epochs see different views of the same interval — a free and very
    effective augmentation for small datasets.
    """

    def __init__(
        self,
        annotations: Sequence[Annotation],
        labels: LabelSpace,
        data: DataConfig,
        task: str = "multilabel",
        training: bool = False,
        cache_path: str | Path | None = None,
        strict: bool = False,
        seed: int = 0,
    ):
        self.labels = labels
        self.data = data
        self.task = task
        self.training = training
        self.seed = seed
        self.transform = ClipTransform(
            size=data.image_size,
            training=training,
            scale=data.train_scale,
            hflip_p=data.hflip if training else 0.0,
            jitter=data.colour_jitter if training else 0.0,
            grey_p=data.grey if training else 0.0,
            erase_p=data.erase if training else 0.0,
        )
        self.skipped: list[tuple[str, str]] = []
        cache = MetaCache(cache_path if data.cache_index else None)
        self.samples: list[Sample] = []

        for ann in annotations:
            try:
                meta = cache.get(ann.video)
            except Exception as exc:
                if strict:
                    raise
                self.skipped.append((ann.video, f"unreadable: {exc}"))
                continue
            try:
                target = self._encode(ann)
            except KeyError as exc:
                if strict:
                    raise
                self.skipped.append((ann.video, str(exc)))
                continue

            first = max(0, int(math.floor(ann.start * meta.fps)))
            last_time = ann.end if ann.end is not None else meta.duration
            last = min(meta.n_frames - 1, int(math.ceil(last_time * meta.fps)) - 1)
            if last < first:
                last = min(meta.n_frames - 1, first)
            if last < 0:
                self.skipped.append((ann.video, "span lies outside the video"))
                continue
            self.samples.append(Sample(ann, meta, first, last, target))
        cache.flush()

        if not self.samples:
            raise ValueError(
                "no usable samples; "
                + (f"{len(self.skipped)} skipped, first: {self.skipped[0]}" if self.skipped else "empty annotations")
            )

    # -- targets ---------------------------------------------------------
    def _encode(self, ann: Annotation) -> np.ndarray:
        if self.task == "multiclass":
            if len(ann.labels) != 1:
                raise KeyError(
                    f"multiclass task needs exactly one label, got {ann.labels!r} "
                    f"for {ann.video}; use task=multilabel for co-occurring behaviours"
                )
            vector = np.zeros(len(self.labels), dtype=np.float32)
            vector[self.labels.index(ann.labels[0])] = 1.0
            return vector
        vector = np.zeros(len(self.labels), dtype=np.float32)
        for name in ann.labels:
            vector[self.labels.index(name)] = 1.0
        return vector

    # -- stats -----------------------------------------------------------
    def label_counts(self) -> Counter:
        counts: Counter = Counter()
        for sample in self.samples:
            for i, on in enumerate(sample.target):
                if on:
                    counts[self.labels.name(i)] += 1
        return counts

    def positives_per_class(self) -> np.ndarray:
        matrix = np.stack([s.target for s in self.samples])
        return matrix.sum(axis=0)

    def groups(self) -> list[str]:
        return [s.annotation.group_key for s in self.samples]

    # -- access ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        # Per-item RNG keyed on (seed, index, epoch-ish entropy) so worker
        # processes do not all draw the same windows.
        rng = np.random.default_rng(
            (self.seed, index, torch.randint(0, 2**31 - 1, (1,)).item())
            if self.training
            else (self.seed, index)
        )
        idx = sample_indices(
            n_frames=sample.n_available,
            clip_frames=self.data.clip_frames,
            stride=self.data.frame_stride,
            mode=self.data.sampling,
            training=self.training,
            rng=rng,
        )
        idx = idx + sample.first_frame
        decode_size = self._decode_size(sample.meta)
        from .video import read_frames  # local import keeps workers light

        clip = read_frames(sample.annotation.video, idx, size=decode_size)
        if self.training and self.data.temporal_reverse and rng.random() < self.data.temporal_reverse:
            clip = clip[::-1]
        tensor = self.transform(clip, rng=rng)
        return {
            "clip": tensor,
            "target": torch.from_numpy(sample.target.copy()),
            "weight": torch.tensor(sample.annotation.weight, dtype=torch.float32),
            "index": index,
        }

    def _decode_size(self, meta: VideoMeta) -> tuple[int, int] | None:
        """Decode at just above the training resolution, preserving aspect ratio.

        This is where most of the CPU time is won: decoding a 1080p clip to
        180x320 instead of 1080x1920 is ~5x less swscale work and 36x less memory,
        and the crop that follows only ever needs ``size/min(scale)`` pixels.
        """
        if meta.width <= 0 or meta.height <= 0:
            return None
        target_short = self.data.image_size
        if self.training:
            # Leave headroom so random-resized-crop still has real pixels to pick.
            target_short = int(round(self.data.image_size / math.sqrt(max(0.05, self.data.train_scale[0]))))
        short = min(meta.width, meta.height)
        if short <= target_short:
            return None  # never upscale during decode
        scale = target_short / short
        width = max(2, int(round(meta.width * scale)))
        height = max(2, int(round(meta.height * scale)))
        return (width - width % 2, height - height % 2)


class SlidingWindowDataset(Dataset):
    """Dense sliding windows over one video — the inference-side dataset."""

    def __init__(
        self,
        video: str | Path,
        data: DataConfig,
        hop_fraction: float = 0.5,
        meta: VideoMeta | None = None,
    ):
        from .video import probe

        self.video = str(video)
        self.data = data
        self.meta = meta or probe(self.video)
        span = data.clip_frames * data.frame_stride
        hop = max(1, int(round(span * max(0.05, hop_fraction))))
        self.span = span
        self.starts = window_starts(self.meta.n_frames, span, hop)
        self.transform = ClipTransform(size=data.image_size, training=False)
        self._decode_size = self._compute_decode_size()

    def _compute_decode_size(self) -> tuple[int, int] | None:
        if self.meta.width <= 0 or self.meta.height <= 0:
            return None
        short = min(self.meta.width, self.meta.height)
        if short <= self.data.image_size:
            return None
        scale = self.data.image_size / short
        w = max(2, int(round(self.meta.width * scale)))
        h = max(2, int(round(self.meta.height * scale)))
        return (w - w % 2, h - h % 2)

    def window_time(self, i: int) -> tuple[float, float]:
        start = self.starts[i]
        end = min(self.meta.n_frames, start + self.span)
        return self.meta.time_of(start), self.meta.time_of(end)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict:
        from .video import read_frames

        start = self.starts[index]
        available = min(self.span, self.meta.n_frames - start)
        idx = sample_indices(
            n_frames=available,
            clip_frames=self.data.clip_frames,
            stride=self.data.frame_stride,
            mode=self.data.sampling,
            training=False,
        ) + start
        clip = read_frames(self.video, idx, size=self._decode_size)
        return {"clip": self.transform(clip), "index": index}


# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------
class RepeatFactorSampler(Sampler[int]):
    """Repeat-factor sampling for long-tailed *multi-label* data (LVIS-style).

    Plain inverse-frequency class balancing does not work for multi-label data: a
    clip labelled ``["running", "barking"]`` cannot be "a barking sample" and "a
    running sample" at once.  Instead we give each class a repeat factor
    ``sqrt(t / f_c)`` and each *sample* the max factor over its labels, so rare
    behaviours are oversampled without distorting the common ones more than
    necessary.  Fractional parts are re-drawn every epoch, so the oversampling is
    stochastic rather than a fixed duplicated list.
    """

    def __init__(
        self,
        dataset: ClipDataset,
        threshold: float = 0.05,
        max_factor: float = 8.0,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        n = len(dataset)
        freq = dataset.positives_per_class() / max(1, n)
        with np.errstate(divide="ignore", invalid="ignore"):
            class_factor = np.sqrt(np.where(freq > 0, threshold / freq, 1.0))
        class_factor = np.clip(class_factor, 1.0, max_factor)
        self.class_factor = class_factor

        factors = np.ones(n, dtype=np.float64)
        for i, sample in enumerate(dataset.samples):
            active = sample.target > 0
            if active.any():
                factors[i] = float(class_factor[active].max())
        self.factors = factors

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng((self.seed, self.epoch))
        integral = np.floor(self.factors).astype(np.int64)
        fractional = self.factors - integral
        extra = (rng.random(len(self.factors)) < fractional).astype(np.int64)
        counts = integral + extra
        indices = np.repeat(np.arange(len(counts)), counts)
        rng.shuffle(indices)
        return iter(indices.tolist())

    def __len__(self) -> int:
        return int(round(self.factors.sum()))


class ClassBalancedSampler(Sampler[int]):
    """Inverse-frequency sampling, for single-label (``multiclass``) training."""

    def __init__(self, dataset: ClipDataset, seed: int = 0):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        counts = dataset.positives_per_class()
        weights = np.zeros(len(dataset), dtype=np.float64)
        for i, sample in enumerate(dataset.samples):
            active = np.flatnonzero(sample.target > 0)
            if len(active) == 0:
                weights[i] = 1.0
                continue
            weights[i] = float(np.mean([1.0 / max(1.0, counts[c]) for c in active]))
        self.weights = weights / weights.sum()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng((self.seed, self.epoch))
        draws = rng.choice(len(self.weights), size=len(self.weights), p=self.weights)
        return iter(draws.tolist())

    def __len__(self) -> int:
        return len(self.weights)


def build_sampler(dataset: ClipDataset, config: Config) -> Sampler | None:
    if not config.train.class_balanced:
        return None
    if config.task == "multilabel":
        return RepeatFactorSampler(dataset, seed=config.train.seed)
    return ClassBalancedSampler(dataset, seed=config.train.seed)


def collate(batch: list[dict]) -> dict:
    return {
        "clip": torch.stack([b["clip"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "weight": torch.stack([b["weight"] for b in batch]),
        "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------
def make_splits(
    annotations: Sequence[Annotation],
    fractions: dict[str, float] | None = None,
    seed: int = 0,
) -> dict[str, list[Annotation]]:
    """Group-aware, label-stratified split.

    Every annotation sharing a ``group_key`` lands in the same split — that is
    the whole point, and it is why validation numbers from this splitter are
    lower (and honest) compared to a naive per-clip shuffle.  Within that
    constraint we greedily assign groups to whichever split is furthest below its
    quota for the group's rarest label, which keeps rare behaviours present in
    every split.
    """
    fractions = fractions or {"train": 0.8, "val": 0.2}
    total = sum(fractions.values())
    if total <= 0:
        raise ValueError("split fractions must sum to a positive number")
    fractions = {k: v / total for k, v in fractions.items()}

    by_group: dict[str, list[Annotation]] = defaultdict(list)
    for ann in annotations:
        by_group[ann.group_key].append(ann)

    label_totals: Counter = Counter()
    for ann in annotations:
        for name in ann.labels:
            label_totals[name] += 1

    def rarity(group: list[Annotation]) -> tuple[int, str]:
        names = {n for a in group for n in a.labels}
        if not names:
            return (10**9, "")
        rarest = min(names, key=lambda n: label_totals[n])
        return (label_totals[rarest], rarest)

    rng = np.random.default_rng(seed)
    groups = list(by_group.items())
    rng.shuffle(groups)
    # Rarest-first: place the hardest-to-balance groups while there is still slack.
    groups.sort(key=lambda kv: rarity(kv[1])[0])

    out: dict[str, list[Annotation]] = {name: [] for name in fractions}
    per_split_labels: dict[str, Counter] = {name: Counter() for name in fractions}
    sizes: Counter = Counter()
    placed = 0

    for _, group in groups:
        _, rarest = rarity(group)
        best, best_score = None, None
        for name, frac in fractions.items():
            if frac <= 0:
                continue
            want_label = frac * max(1, label_totals[rarest]) if rarest else 0.0
            have_label = per_split_labels[name][rarest] if rarest else 0.0
            label_deficit = (want_label - have_label) / max(1.0, want_label or 1.0)
            size_deficit = frac - (sizes[name] / placed if placed else 0.0)
            score = label_deficit * 2.0 + size_deficit
            if best_score is None or score > best_score:
                best, best_score = name, score
        assert best is not None
        out[best].extend(group)
        sizes[best] += len(group)
        placed += len(group)
        for ann in group:
            for name in ann.labels:
                per_split_labels[best][name] += 1
    return out
