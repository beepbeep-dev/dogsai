"""Decode-once clip cache.

Video training is almost always decode-bound rather than compute-bound. On real
footage this package measured ~2.5 clips/s end to end on four CPU cores — and of
that, the model forward and backward pass was a small minority. Every epoch was
re-decoding the same H.264 bitstreams to produce the same pixels.

So decode each annotated span once, into a compact fixed-size ``uint8`` array, and
train off that. For 1217 clips at 32 frames of 160x160 the cache is ~3 GB, which
memory-maps comfortably and turns epoch time from minutes into seconds.

What is preserved and what is given up
--------------------------------------
The cache stores *more* frames than a clip needs (``cache_frames`` >=
``clip_frames``), so training still gets:

* **temporal jitter** — a different subset of the cached frames each epoch;
* **spatial augmentation** — the cache is stored at a margin above the training
  resolution, so random-resized-crop still has real pixels to choose from;
* flips, colour jitter, erasing and mixup, all unchanged.

What is given up: crops at the source resolution, and temporal offsets finer than
the cached stride. That is a real reduction in augmentation diversity, and on a
large dataset with plenty of epochs the uncached path is still the better one.
On anything small enough that decode dominates — which is most animal-behaviour
datasets — the trade is strongly worth it.

The cache is keyed by the parameters that determine its contents, so changing the
resolution or frame count invalidates it rather than silently training on a
mismatched array.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DataConfig
from .dataset import Annotation
from .labels import LabelSpace
from .transforms import ClipTransform
from .video import MetaCache, probe, read_frames, sample_indices

CACHE_VERSION = 1


@dataclass(frozen=True)
class CacheSpec:
    """The parameters that define a cache's contents."""

    frames: int = 32
    size: int = 176
    stride_mode: str = "span"

    def key(self) -> dict:
        return {
            "version": CACHE_VERSION,
            "frames": self.frames,
            "size": self.size,
            "stride_mode": self.stride_mode,
        }


def _decode_one(job: tuple[int, str, float, float | None, int, int]) -> tuple[int, np.ndarray | None, str]:
    """Worker: decode one span into ``(frames, size, size, 3)`` uint8.

    Runs in a separate process, so it must not touch shared state and must return
    its index for reassembly.
    """
    index, video, start, end, frames, size = job
    try:
        meta = probe(video)
        first = max(0, int(math.floor(start * meta.fps)))
        last_time = end if end is not None else meta.duration
        last = min(meta.n_frames - 1, int(math.ceil(last_time * meta.fps)) - 1)
        if last < first:
            last = min(meta.n_frames - 1, first)
        available = last - first + 1
        # Spread the cached frames across the whole span: this is the pool that
        # per-epoch temporal jitter will later sample from.
        idx = sample_indices(available, frames, stride=1, mode="tsn", training=False) + first

        # Decode with the short side at `size`, then centre-crop to square. The
        # crop is deliberate: storing a non-square array would mean either padding
        # or a per-item aspect ratio, and both complicate the memmap for no gain.
        clip = read_frames(video, idx, size=_decode_size(meta.width, meta.height, size))
        clip = _centre_square(clip, size)
        return index, clip, ""
    except Exception as exc:
        return index, None, f"{type(exc).__name__}: {exc}"


def _decode_size(width: int, height: int, size: int) -> tuple[int, int] | None:
    if width <= 0 or height <= 0:
        return None
    short = min(width, height)
    if short <= size:
        return None
    scale = size / short
    w = max(2, int(round(width * scale)))
    h = max(2, int(round(height * scale)))
    return (w - w % 2, h - h % 2)


def _centre_square(clip: np.ndarray, size: int) -> np.ndarray:
    """Resize short side to ``size`` if needed, then centre-crop to ``size``."""
    from .transforms import _resize_clip

    _, h, w, _ = clip.shape
    if min(h, w) != size:
        scale = size / min(h, w)
        new_w = max(size, int(round(w * scale)))
        new_h = max(size, int(round(h * scale)))
        clip = _resize_clip(clip, new_w, new_h)
        _, h, w, _ = clip.shape
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return np.ascontiguousarray(clip[:, y0 : y0 + size, x0 : x0 + size])


def build_cache(
    annotations: Sequence[Annotation],
    out_dir: str | Path,
    spec: CacheSpec | None = None,
    workers: int = 4,
    verbose: bool = True,
) -> Path:
    """Decode every annotation into a memory-mappable array on disk.

    Writes ``clips.npy`` (the pixels) and ``index.json`` (labels, groups, spec).
    Decoding is parallelised across processes because it is CPU-bound in ffmpeg
    rather than in Python.
    """
    spec = spec or CacheSpec()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_path = out_dir / "clips.npy"
    index_path = out_dir / "index.json"

    n = len(annotations)
    if n == 0:
        raise ValueError("nothing to cache")
    shape = (n, spec.frames, spec.size, spec.size, 3)
    nbytes = int(np.prod(shape))
    if verbose:
        print(f"caching {n} clips -> {clips_path} ({nbytes / 1e9:.2f} GB)")

    array = np.lib.format.open_memmap(clips_path, mode="w+", dtype=np.uint8, shape=shape)
    jobs = [
        (i, a.video, a.start, a.end, spec.frames, spec.size)
        for i, a in enumerate(annotations)
    ]
    failures: list[tuple[int, str]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for index, clip, error in pool.map(_decode_one, jobs, chunksize=4):
            done += 1
            if clip is None:
                failures.append((index, error))
            else:
                array[index] = clip
            if verbose and done % 100 == 0:
                print(f"  {done}/{n}", flush=True)
    array.flush()

    ok = [i for i in range(n) if i not in {f[0] for f in failures}]
    payload = {
        "spec": spec.key(),
        "count": n,
        "usable": ok,
        "records": [
            {
                "video": a.video,
                "labels": a.labels,
                "start": a.start,
                "end": a.end,
                "group": a.group_key,
                "weight": a.weight,
            }
            for a in annotations
        ],
        "failures": [{"index": i, "error": e} for i, e in failures],
    }
    index_path.write_text(json.dumps(payload))
    if verbose:
        print(f"  cached {len(ok)}/{n} clips" + (f", {len(failures)} failed" if failures else ""))
    return out_dir


def cache_is_valid(cache_dir: str | Path, spec: CacheSpec) -> bool:
    """Whether an existing cache matches ``spec`` and is complete."""
    cache_dir = Path(cache_dir)
    index_path = cache_dir / "index.json"
    clips_path = cache_dir / "clips.npy"
    if not index_path.exists() or not clips_path.exists():
        return False
    try:
        payload = json.loads(index_path.read_text())
    except Exception:
        return False
    return payload.get("spec") == spec.key() and bool(payload.get("usable"))


class CachedClipDataset(Dataset):
    """Trains from a :func:`build_cache` directory instead of decoding video.

    Temporal jitter is preserved by sampling ``clip_frames`` positions out of the
    ``cache_frames`` stored per clip — random while training, evenly spaced while
    evaluating.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        labels: LabelSpace,
        data: DataConfig,
        task: str = "multilabel",
        training: bool = False,
        seed: int = 0,
    ):
        self.cache_dir = Path(cache_dir)
        payload = json.loads((self.cache_dir / "index.json").read_text())
        self.spec = CacheSpec(**{k: v for k, v in payload["spec"].items() if k != "version"})
        self.clips = np.load(self.cache_dir / "clips.npy", mmap_mode="r")
        self.labels = labels
        self.data = data
        self.task = task
        self.training = training
        self.seed = seed
        self.skipped: list[tuple[str, str]] = [
            (payload["records"][f["index"]]["video"], f["error"]) for f in payload.get("failures", [])
        ]

        if data.image_size > self.spec.size:
            raise ValueError(
                f"cache holds {self.spec.size}px clips but training wants "
                f"{data.image_size}px; rebuild the cache with a larger --cache-size"
            )
        if data.clip_frames > self.spec.frames:
            raise ValueError(
                f"cache holds {self.spec.frames} frames per clip but training wants "
                f"{data.clip_frames}; rebuild with a larger --cache-frames"
            )

        self.transform = ClipTransform(
            size=data.image_size,
            training=training,
            scale=data.train_scale,
            hflip_p=data.hflip if training else 0.0,
            jitter=data.colour_jitter if training else 0.0,
            grey_p=data.grey if training else 0.0,
            erase_p=data.erase if training else 0.0,
        )

        self.indices: list[int] = []
        self.targets: list[np.ndarray] = []
        self.weights: list[float] = []
        self.group_keys: list[str] = []
        for i in payload["usable"]:
            record = payload["records"][i]
            try:
                target = self._encode(record["labels"])
            except KeyError as exc:
                self.skipped.append((record["video"], str(exc)))
                continue
            self.indices.append(i)
            self.targets.append(target)
            self.weights.append(float(record.get("weight", 1.0)))
            self.group_keys.append(record.get("group") or record["video"])
        if not self.indices:
            raise ValueError(f"cache at {cache_dir} yielded no usable samples")

    def _encode(self, names: Sequence[str]) -> np.ndarray:
        vector = np.zeros(len(self.labels), dtype=np.float32)
        if self.task == "multiclass":
            if len(names) != 1:
                raise KeyError(f"multiclass needs exactly one label, got {list(names)}")
            vector[self.labels.index(names[0])] = 1.0
            return vector
        for name in names:
            vector[self.labels.index(name)] = 1.0
        return vector

    # -- parity with ClipDataset so samplers and the Trainer work unchanged ---
    def positives_per_class(self) -> np.ndarray:
        return np.stack(self.targets).sum(axis=0)

    def label_counts(self):
        from collections import Counter

        counts: Counter = Counter()
        for target in self.targets:
            for i, on in enumerate(target):
                if on:
                    counts[self.labels.name(i)] += 1
        return counts

    def groups(self) -> list[str]:
        return list(self.group_keys)

    @property
    def samples(self) -> list:
        """Minimal shim: samplers only read ``.target`` off each entry."""
        return [_CachedSample(t) for t in self.targets]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict:
        row = self.indices[index]
        cached = np.asarray(self.clips[row])  # (cache_frames, S, S, 3)
        rng = np.random.default_rng(
            (self.seed, index, int(torch.randint(0, 2**31 - 1, (1,)).item()))
            if self.training
            else (self.seed, index)
        )
        pick = sample_indices(
            n_frames=self.spec.frames,
            clip_frames=self.data.clip_frames,
            stride=1,
            mode="tsn",
            training=self.training,
            rng=rng,
        )
        clip = cached[pick]
        if self.training and self.data.temporal_reverse and rng.random() < self.data.temporal_reverse:
            clip = clip[::-1]
        return {
            "clip": self.transform(clip, rng=rng),
            "target": torch.from_numpy(self.targets[index].copy()),
            "weight": torch.tensor(self.weights[index], dtype=torch.float32),
            "index": index,
        }


@dataclass
class _CachedSample:
    target: np.ndarray


def build_split_caches(
    splits: dict[str, Sequence[Annotation]],
    root: str | Path,
    spec: CacheSpec | None = None,
    workers: int = 4,
    force: bool = False,
    verbose: bool = True,
) -> dict[str, Path]:
    """Build (or reuse) a cache per split under ``root/.cache/<split>``."""
    spec = spec or CacheSpec()
    root = Path(root)
    out: dict[str, Path] = {}
    for name, annotations in splits.items():
        cache_dir = root / ".cache" / name
        if not force and cache_is_valid(cache_dir, spec):
            if verbose:
                print(f"reusing cache for {name}: {cache_dir}")
            out[name] = cache_dir
            continue
        out[name] = build_cache(annotations, cache_dir, spec, workers=workers, verbose=verbose)
    return out


def estimate_cache_size(n_clips: int, spec: CacheSpec | None = None) -> float:
    """Cache size in GB, for deciding whether it fits before writing it."""
    spec = spec or CacheSpec()
    return n_clips * spec.frames * spec.size * spec.size * 3 / 1e9


def warm_meta_cache(annotations: Sequence[Annotation], cache_path: str | Path) -> None:
    """Probe every video once so later passes are O(1)."""
    cache = MetaCache(cache_path)
    for annotation in annotations:
        try:
            cache.get(annotation.video)
        except Exception:
            continue
    cache.flush()
