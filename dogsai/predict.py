"""Inference: an mp4 in, a behaviour timeline out.

A classifier gives you a score per window; what a user wants is "the dog was
digging from 4.2s to 9.8s".  Getting from one to the other is where most of the
perceived accuracy of a video model lives, so the post-processing here is
deliberate rather than incidental:

1. **Overlapping windows** (default 50% hop) so a behaviour straddling a boundary
   is still seen whole by at least one window.
2. **Median smoothing** across windows.  A median rejects the single-window
   outlier that a mean would happily average in — that outlier is exactly what
   produces the "flickering label" failure mode.
3. **Per-class thresholds** fitted on validation data and carried in the
   checkpoint, rather than a hardcoded 0.5.
4. **Span hysteresis** — merge same-behaviour spans across short gaps, then drop
   spans shorter than ``min_duration``.  Together these turn a noisy binary mask
   into intervals a human would agree with.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .affect import AffectReading
from .config import Config, InferenceConfig
from .dataset import SlidingWindowDataset
from .engine import load_checkpoint
from .labels import AROUSAL_FLAGS, BehaviourSpan, LabelSpace
from .model import DogBehaviourNet
from .video import VideoMeta


def median_smooth(scores: np.ndarray, width: int) -> np.ndarray:
    """Sliding median along the window axis, edges handled by clamping."""
    if width <= 1 or len(scores) < 3:
        return scores
    width = min(width, len(scores))
    if width % 2 == 0:
        width += 1
    half = width // 2
    padded = np.pad(scores, ((half, half), (0, 0)), mode="edge")
    strided = np.lib.stride_tricks.sliding_window_view(padded, width, axis=0)
    return np.median(strided, axis=-1)


def spans_from_mask(
    mask: np.ndarray,
    scores: np.ndarray,
    times: list[tuple[float, float]],
    behaviour: str,
    merge_gap: float = 0.0,
    min_duration: float = 0.0,
) -> list[BehaviourSpan]:
    """Convert a per-window boolean mask into merged, filtered spans."""
    spans: list[BehaviourSpan] = []
    start_index: int | None = None
    for i, active in enumerate(list(mask) + [False]):
        if active and start_index is None:
            start_index = i
        elif not active and start_index is not None:
            window = scores[start_index:i]
            spans.append(
                BehaviourSpan(
                    behaviour=behaviour,
                    start=times[start_index][0],
                    end=times[i - 1][1],
                    score=float(window.mean()),
                    peak=float(window.max()),
                )
            )
            start_index = None

    if merge_gap > 0 and spans:
        merged = [spans[0]]
        for span in spans[1:]:
            previous = merged[-1]
            if span.start - previous.end <= merge_gap:
                total = previous.duration + span.duration
                if total > 0:
                    previous.score = (
                        previous.score * previous.duration + span.score * span.duration
                    ) / total
                previous.peak = max(previous.peak, span.peak)
                previous.end = span.end
            else:
                merged.append(span)
        spans = merged

    return [s for s in spans if s.duration >= min_duration]


@dataclass
class VideoPrediction:
    """The full result for one video."""

    video: str
    meta: VideoMeta
    behaviours: list[str]
    task: str
    window_times: list[tuple[float, float]]
    scores: np.ndarray  # (n_windows, n_classes), post-smoothing
    spans: list[BehaviourSpan] = field(default_factory=list)
    thresholds: np.ndarray | None = None

    # -- aggregates ------------------------------------------------------
    def time_per_behaviour(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for span in self.spans:
            totals[span.behaviour] = totals.get(span.behaviour, 0.0) + span.duration
        return dict(sorted(totals.items(), key=lambda kv: -kv[1]))

    def dominant(self) -> str | None:
        totals = self.time_per_behaviour()
        return next(iter(totals), None)

    def arousal_fraction(self) -> float:
        """Fraction of the video covered by commonly stress-associated behaviours.

        A screening aid for "which of these 400 clips should a human watch", not a
        welfare assessment — the taxonomy has no way to distinguish an excited
        bark from a distressed one.
        """
        if self.meta.duration <= 0:
            return 0.0
        flagged = sum(
            span.duration for span in self.spans if span.behaviour in AROUSAL_FLAGS
        )
        return min(1.0, flagged / self.meta.duration)

    def score_at(self, seconds: float) -> dict[str, float]:
        """Interpolate the per-class scores at an arbitrary timestamp."""
        if not self.window_times:
            return {}
        centres = np.array([(a + b) / 2 for a, b in self.window_times])
        i = int(np.argmin(np.abs(centres - seconds)))
        return {name: float(self.scores[i, c]) for c, name in enumerate(self.behaviours)}

    def affect(self) -> AffectReading:
        """A body-language read derived from the detected spans.

        See :mod:`dogsai.affect` for what this can and cannot tell you — briefly:
        it estimates arousal and valence from observable behaviour, it is not a
        measurement of emotion, and it is capped by what the model was trained to
        recognise.
        """
        from .affect import read_affect

        return read_affect(self.spans, self.meta.duration, known_behaviours=self.behaviours)

    def to_dict(self) -> dict:
        return {
            "video": self.video,
            "duration": round(self.meta.duration, 3),
            "fps": round(self.meta.fps, 3),
            "resolution": [self.meta.width, self.meta.height],
            "task": self.task,
            "behaviours": self.behaviours,
            "n_windows": len(self.window_times),
            "dominant": self.dominant(),
            "time_per_behaviour": {k: round(v, 3) for k, v in self.time_per_behaviour().items()},
            "arousal_fraction": round(self.arousal_fraction(), 4),
            "spans": [span.to_dict() for span in self.spans],
            "affect": self.affect().to_dict(),
        }

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    def timeline(self, width: int = 56) -> str:
        """ASCII timeline: one row per behaviour that ever fires."""
        active = [b for b in self.behaviours if any(s.behaviour == b for s in self.spans)]
        if not active:
            return "(no behaviours above threshold)"
        duration = max(self.meta.duration, 1e-6)
        label_width = max(len(b) for b in active) + 1
        lines = []
        for name in active:
            row = [" "] * width
            for span in self.spans:
                if span.behaviour != name:
                    continue
                a = int(span.start / duration * width)
                b = max(a + 1, int(span.end / duration * width))
                for i in range(max(0, a), min(width, b)):
                    row[i] = "#" if span.score > 0.75 else "="
            total = self.time_per_behaviour().get(name, 0.0)
            lines.append(f"{name.ljust(label_width)}|{''.join(row)}| {total:5.1f}s")
        ruler = " " * label_width + f"0s{'-' * (width - 8)}{duration:5.1f}s"
        return "\n".join(lines + [ruler])

    def __str__(self) -> str:
        head = f"{self.video}  ({self.meta.duration:.1f}s, {len(self.window_times)} windows)"
        body = "\n".join(str(s) for s in self.spans) or "  no behaviours detected"
        return f"{head}\n{body}"


class BehaviourPredictor:
    """Loads a checkpoint once, then annotates any number of videos."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device | None = None,
        infer: InferenceConfig | None = None,
        prefer_ema: bool = True,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.config, self.labels, self.extra = load_checkpoint(
            checkpoint, device=self.device, prefer_ema=prefer_ema
        )
        if infer is not None:
            self.config.infer = infer
        saved = self.extra.get("thresholds")
        if saved:
            self.thresholds = np.asarray(saved, dtype=np.float64)
        elif self.config.task == "multiclass":
            # A softmax winner over C classes rarely exceeds 0.5, so the
            # multilabel default of 0.5 would suppress essentially every
            # prediction.  The principled floor is "beats a uniform guess".
            self.thresholds = np.full(len(self.labels), 1.0 / len(self.labels))
        else:
            self.thresholds = np.full(len(self.labels), self.config.infer.threshold)

    # -- core ------------------------------------------------------------
    @torch.no_grad()
    def raw_scores(self, video: str | Path) -> tuple[np.ndarray, SlidingWindowDataset]:
        cfg = self.config.infer
        dataset = SlidingWindowDataset(
            video, self.config.data, hop_fraction=cfg.window_stride
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=min(2, self.config.data.num_workers),
            collate_fn=lambda batch: torch.stack([b["clip"] for b in batch]),
        )
        chunks = []
        for clips in loader:
            clips = clips.to(self.device, non_blocking=True)
            logits = self.model(clips).float()
            if cfg.tta_hflip:
                # Horizontal flip is the only label-preserving TTA here; a
                # temporal flip would change what some behaviours mean.
                logits = (logits + self.model(clips.flip(-1)).float()) / 2
            chunks.append(logits.cpu().numpy())
        logits = np.concatenate(chunks) if chunks else np.zeros((0, len(self.labels)))
        if self.config.task == "multiclass":
            shifted = logits - logits.max(axis=1, keepdims=True)
            exponent = np.exp(shifted)
            scores = exponent / np.maximum(exponent.sum(axis=1, keepdims=True), 1e-12)
        else:
            scores = 1.0 / (1.0 + np.exp(-logits))
        return scores, dataset

    def predict(self, video: str | Path) -> VideoPrediction:
        cfg = self.config.infer
        scores, dataset = self.raw_scores(video)
        scores = median_smooth(scores, cfg.smooth)
        times = [dataset.window_time(i) for i in range(len(dataset))]

        spans: list[BehaviourSpan] = []
        if self.config.task == "multiclass":
            # One behaviour at a time: take the argmax per window, then segment.
            winners = scores.argmax(axis=1)
            confidence = scores.max(axis=1)
            for c, name in enumerate(self.labels):
                mask = (winners == c) & (confidence >= self.thresholds[c])
                spans += spans_from_mask(
                    mask, scores[:, c], times, name, cfg.merge_gap, cfg.min_duration
                )
        else:
            for c, name in enumerate(self.labels):
                mask = scores[:, c] >= self.thresholds[c]
                spans += spans_from_mask(
                    mask, scores[:, c], times, name, cfg.merge_gap, cfg.min_duration
                )
        spans.sort(key=lambda s: (s.start, s.behaviour))

        return VideoPrediction(
            video=str(video),
            meta=dataset.meta,
            behaviours=self.labels.to_list(),
            task=self.config.task,
            window_times=times,
            scores=scores,
            spans=spans,
            thresholds=self.thresholds,
        )

    def predict_many(self, videos: list[str | Path]) -> list[VideoPrediction]:
        out = []
        for video in videos:
            try:
                out.append(self.predict(video))
            except Exception as exc:
                print(f"  ! {video}: {exc}")
        return out

    @torch.no_grad()
    def explain(self, video: str | Path, at: float = 0.0) -> dict:
        """Per-frame attention weights for the window containing ``at``.

        Shows *which frames* of the clip drove the prediction — a cheap sanity
        check that the model is looking at the behaviour and not at, say, the
        first frame's background.
        """
        dataset = SlidingWindowDataset(
            video, self.config.data, hop_fraction=self.config.infer.window_stride
        )
        centres = [sum(dataset.window_time(i)) / 2 for i in range(len(dataset))]
        index = int(np.argmin(np.abs(np.array(centres) - at)))
        clip = dataset[index]["clip"].unsqueeze(0).to(self.device)
        logits, attention = self.model(clip, return_attention=True)
        scores = (
            torch.softmax(logits, dim=-1) if self.config.task == "multiclass" else torch.sigmoid(logits)
        )[0].cpu().numpy()
        top = np.argsort(-scores)[:5]
        return {
            "window": dataset.window_time(index),
            "top": [
                {"behaviour": self.labels.name(int(c)), "score": float(scores[c])} for c in top
            ],
            "frame_attention": attention[0].cpu().numpy().tolist(),
        }


def render_overlay(
    prediction: VideoPrediction,
    out_path: str | Path,
    max_labels: int = 4,
) -> Path:
    """Burn the detected behaviours into a copy of the video.

    Requires OpenCV for the text drawing; the encode goes through PyAV so the
    output is a normal H.264 mp4.
    """
    import cv2

    from .video import read_frames, write_video

    meta = prediction.meta
    spans_by_time = prediction.spans

    def frames():
        chunk = 64
        for start in range(0, meta.n_frames, chunk):
            indices = np.arange(start, min(meta.n_frames, start + chunk))
            batch = read_frames(prediction.video, indices)
            for offset, frame in zip(indices, batch):
                t = meta.time_of(int(offset))
                active = [s for s in spans_by_time if s.start <= t <= s.end]
                active.sort(key=lambda s: -s.score)
                canvas = np.ascontiguousarray(frame)
                overlay = canvas.copy()
                height = 26 * min(len(active), max_labels) + 12
                if active:
                    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], height), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
                for row, span in enumerate(active[:max_labels]):
                    cv2.putText(
                        canvas,
                        f"{span.behaviour} {span.score:.2f}",
                        (10, 24 + row * 26),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (80, 255, 120) if span.score > 0.75 else (120, 200, 255),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    canvas,
                    f"{t:6.2f}s",
                    (10, canvas.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )
                yield canvas

    return write_video(out_path, frames(), fps=meta.fps)


def load_predictor(checkpoint: str | Path, **kwargs) -> BehaviourPredictor:
    return BehaviourPredictor(checkpoint, **kwargs)
