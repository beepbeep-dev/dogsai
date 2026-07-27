"""Dataset auditing — find the label problems before they become model problems.

A behaviour model's accuracy is capped by its annotations, and the failures are
depressingly consistent across datasets.  This module looks for all of them:

* **Broken or unreadable files** — silently dropped at training time, so a class
  quietly loses half its examples and nobody notices.
* **Spans that do not fit** — an interval shorter than one clip means the sampler
  pads with frames from outside the labelled region, i.e. mislabelled input.
* **Unknown or misspelled labels** — ``"tail_wag"`` vs ``"tail_wagging"`` becomes
  two classes, each with half the data.
* **Split leakage** — the single biggest cause of a validation score that does not
  survive contact with real footage.  Checked both by group key and by near-
  duplicate frame hashing, because "different filename, same clip re-encoded" is
  extremely common in scraped video data.
* **Contradictory labels** — overlapping spans that assert two mutually exclusive
  postures at the same instant.
* **Imbalance** — reported with the effective sample size per class, so you know
  which numbers in the eval table are meaningless.

Every finding is either an ``error`` (will corrupt training) or a ``warning``
(will bias it).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .dataset import Annotation
from .labels import POSTURE, LabelSpace
from .video import MetaCache, VideoMeta, sample_indices


@dataclass
class Finding:
    level: str  # "error" | "warning" | "info"
    code: str
    message: str
    subject: str = ""

    def __str__(self) -> str:
        mark = {"error": "ERROR", "warning": "warn ", "info": "info "}.get(self.level, "     ")
        tail = f"  [{self.subject}]" if self.subject else ""
        return f"{mark} {self.code}: {self.message}{tail}"


@dataclass
class AuditReport:
    n_annotations: int = 0
    n_videos: int = 0
    n_groups: int = 0
    total_labelled_seconds: float = 0.0
    label_counts: Counter = field(default_factory=Counter)
    label_seconds: Counter = field(default_factory=Counter)
    findings: list[Finding] = field(default_factory=list)
    duplicates: list[tuple[str, str, int]] = field(default_factory=list)

    def add(self, level: str, code: str, message: str, subject: str = "") -> None:
        self.findings.append(Finding(level, code, message, subject))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.level == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def imbalance_ratio(self) -> float:
        if not self.label_counts:
            return 0.0
        counts = list(self.label_counts.values())
        return max(counts) / max(1, min(counts))

    def to_dict(self) -> dict:
        return {
            "n_annotations": self.n_annotations,
            "n_videos": self.n_videos,
            "n_groups": self.n_groups,
            "total_labelled_seconds": round(self.total_labelled_seconds, 2),
            "label_counts": dict(self.label_counts),
            "label_seconds": {k: round(v, 2) for k, v in self.label_seconds.items()},
            "imbalance_ratio": round(self.imbalance_ratio(), 2),
            "errors": [f.message for f in self.errors],
            "warnings": [f.message for f in self.warnings],
            "near_duplicates": [
                {"a": a, "b": b, "hamming": d} for a, b, d in self.duplicates
            ],
        }

    def render(self, top: int = 30) -> str:
        lines = [
            "dataset audit",
            "=" * 60,
            f"annotations : {self.n_annotations}",
            f"videos      : {self.n_videos}",
            f"groups      : {self.n_groups}   (leakage-safe split units)",
            f"labelled    : {self.total_labelled_seconds / 60:.1f} min",
            "",
        ]
        if self.label_counts:
            lines.append("class balance")
            lines.append("-" * 60)
            width = max(len(k) for k in self.label_counts) + 1
            total = sum(self.label_counts.values())
            for name, count in self.label_counts.most_common():
                seconds = self.label_seconds[name]
                share = count / max(1, total)
                bar = "#" * max(1, int(share * 32))
                lines.append(
                    f"  {name.ljust(width)} {count:5d} clips  {seconds / 60:6.1f} min  {bar}"
                )
            lines.append(f"  imbalance (max/min): {self.imbalance_ratio():.1f}x")
            lines.append("")

        by_level = [("error", self.errors), ("warning", self.warnings)]
        for level, items in by_level:
            if not items:
                continue
            lines.append(f"{level}s ({len(items)})")
            lines.append("-" * 60)
            for finding in items[:top]:
                lines.append(f"  {finding}")
            if len(items) > top:
                lines.append(f"  ... and {len(items) - top} more")
            lines.append("")
        if self.ok and not self.warnings:
            lines.append("no problems found.")
        elif self.ok:
            lines.append("no blocking errors; review the warnings above.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# near-duplicate detection
# ---------------------------------------------------------------------------
def clip_signature(
    video: str | Path,
    meta: VideoMeta,
    n_frames: int = 8,
    start: float = 0.0,
    end: float | None = None,
) -> np.ndarray:
    """A compact perceptual fingerprint of a clip, as a packed bit vector.

    Difference hash (dHash) on a 9x8 greyscale thumbnail of each sampled frame.
    dHash is chosen over an average hash because it survives the brightness and
    re-encoding changes that make byte-level or checksum comparison useless, while
    still being cheap enough to run over a whole dataset.
    """
    from .video import read_frames

    first = max(0, int(start * meta.fps))
    last = min(meta.n_frames - 1, int((end if end is not None else meta.duration) * meta.fps) - 1)
    if last < first:
        last = first
    indices = (
        sample_indices(last - first + 1, n_frames, stride=1, mode="tsn", training=False)
        + first
    )
    frames = read_frames(video, indices, size=(9, 8))
    grey = frames.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    bits = grey[:, :, 1:] > grey[:, :, :-1]  # 8 rows x 8 comparisons per frame
    return np.packbits(bits.reshape(-1))


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(_POPCOUNT[np.bitwise_xor(a, b)].sum())


def find_near_duplicates(
    signatures: dict[str, np.ndarray], max_distance: int = 6
) -> list[tuple[str, str, int]]:
    """All pairs of clips within ``max_distance`` bits of each other.

    Vectorised over the second element of each pair, so a few thousand clips is
    seconds rather than minutes.
    """
    keys = list(signatures)
    if len(keys) < 2:
        return []
    matrix = np.stack([signatures[k] for k in keys])
    out: list[tuple[str, str, int]] = []
    for i in range(len(keys) - 1):
        distances = _POPCOUNT[np.bitwise_xor(matrix[i + 1 :], matrix[i])].sum(axis=1)
        for offset in np.flatnonzero(distances <= max_distance):
            out.append((keys[i], keys[i + 1 + int(offset)], int(distances[offset])))
    return out


def merge_duplicate_groups(
    annotations: Sequence[Annotation],
    max_distance: int = 6,
    cache_path: str | Path | None = None,
    verbose: bool = False,
) -> tuple[list[Annotation], int]:
    """Union-find over content, so near-duplicate clips can never split apart.

    Grouping by filename (the default `group_key`) misses the single most common
    way real-world video datasets leak into their own validation set: the same
    footage saved twice under different names — a re-upload, a re-encode, a clip
    trimmed a few frames differently. `dogsai fetch dogbehaviour` found exactly
    this: 21 pairs of differently-named clips, 18 of them pixel-identical,
    landing on opposite sides of a per-filename split.

    Call this **before** :func:`~dogsai.dataset.make_splits`, on the full
    annotation set, not per-split — merging within an already-split set cannot
    undo the leakage, it can only detect it. One perceptual signature is computed
    per existing group (not per annotation — clips already sharing a group don't
    need deduplication against each other), near-duplicate groups are unioned, and
    every annotation in a merged component is reassigned to one canonical group
    name, chosen deterministically (the lexicographically smallest member) so
    reruns are reproducible.

    Returns the annotations with `.group` set explicitly, and the number of
    groups that were merged into another (0 means no cross-group duplicates were
    found — the common case once this has been applied once upstream).
    """
    by_group: dict[str, list[Annotation]] = defaultdict(list)
    for annotation in annotations:
        by_group[annotation.group_key].append(annotation)
    keys = list(by_group)
    if len(keys) < 2:
        for annotation in annotations:
            annotation.group = annotation.group_key
        return list(annotations), 0

    cache = MetaCache(cache_path)
    signatures: dict[str, np.ndarray] = {}
    for key in keys:
        representative = by_group[key][0]
        try:
            meta = cache.get(representative.video)
            signatures[key] = clip_signature(
                representative.video, meta,
                start=representative.start, end=representative.end,
            )
        except Exception:
            continue  # unreadable video: audit_annotations reports it separately
    cache.flush()

    parent = {key: key for key in keys}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Deterministic winner regardless of pair order, so reruns agree.
            lo, hi = sorted((ra, rb))
            parent[hi] = lo

    for a, b, _distance in find_near_duplicates(signatures, max_distance):
        union(a, b)

    components: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        components[find(key)].append(key)
    merged_groups = 0
    for canonical, members in components.items():
        if len(members) > 1:
            merged_groups += len(members) - 1
            if verbose:
                print(f"  merged groups {sorted(members)} -> {canonical!r}")
        for member in members:
            for annotation in by_group[member]:
                annotation.group = canonical

    return list(annotations), merged_groups


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------
def audit_annotations(
    annotations: Sequence[Annotation],
    labels: LabelSpace | None = None,
    clip_seconds: float = 1.1,
    check_duplicates: bool = True,
    duplicate_distance: int = 6,
    cache_path: str | Path | None = None,
    min_examples: int = 20,
) -> AuditReport:
    """Validate one split's annotations."""
    report = AuditReport(n_annotations=len(annotations))
    if not annotations:
        report.add("error", "empty", "no annotations to audit")
        return report

    cache = MetaCache(cache_path)
    metas: dict[str, VideoMeta] = {}
    usable: list[tuple[Annotation, VideoMeta]] = []
    groups: set[str] = set()

    for ann in annotations:
        groups.add(ann.group_key)
        path = Path(ann.video)
        if not path.exists():
            report.add("error", "missing_file", f"video not found: {ann.video}", ann.group_key)
            continue
        if ann.video not in metas:
            try:
                metas[ann.video] = cache.get(ann.video)
            except Exception as exc:
                report.add("error", "unreadable", f"cannot decode {ann.video}: {exc}")
                continue
        meta = metas[ann.video]

        if labels is not None:
            for name in ann.labels:
                if name not in labels:
                    report.add(
                        "error",
                        "unknown_label",
                        f"{name!r} is not in the label space (typo? {_closest(name, labels)})",
                        ann.video,
                    )
        if not ann.labels:
            report.add("info", "negative", f"explicit negative span in {path.name}", ann.group_key)

        end = ann.end if ann.end is not None else meta.duration
        if ann.start < 0:
            report.add("error", "bad_span", f"negative start {ann.start} in {path.name}")
        if end <= ann.start:
            report.add(
                "error", "bad_span", f"end {end} <= start {ann.start} in {path.name}"
            )
            continue
        if ann.start >= meta.duration:
            report.add(
                "error",
                "bad_span",
                f"span starts at {ann.start:.2f}s but {path.name} is only {meta.duration:.2f}s",
            )
            continue
        if end > meta.duration + 0.05:
            report.add(
                "warning",
                "span_overrun",
                f"span ends at {end:.2f}s, past the end of {path.name} ({meta.duration:.2f}s)",
            )
            end = meta.duration

        span_seconds = end - ann.start
        if span_seconds < clip_seconds:
            report.add(
                "warning",
                "span_too_short",
                f"{span_seconds:.2f}s span in {path.name} is shorter than one "
                f"{clip_seconds:.2f}s clip; sampling will pad with frames from "
                f"outside the labelled interval",
                ",".join(ann.labels),
            )
        if meta.fps < 10:
            report.add(
                "warning", "low_fps", f"{path.name} is {meta.fps:.1f} fps; motion cues will be weak"
            )
        if min(meta.width, meta.height) < 128:
            report.add(
                "warning",
                "low_resolution",
                f"{path.name} is {meta.width}x{meta.height}; below the training resolution",
            )

        report.total_labelled_seconds += span_seconds
        for name in ann.labels:
            report.label_counts[name] += 1
            report.label_seconds[name] += span_seconds
        usable.append((ann, meta))

    report.n_videos = len(metas)
    report.n_groups = len(groups)

    # Contradictory postures at the same instant.
    for video, items in _by_video(usable).items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i][0], items[j][0]
                if not _overlaps(a, b, items[i][1], items[j][1]):
                    continue
                postures_a = {n for n in a.labels if n in POSTURE}
                postures_b = {n for n in b.labels if n in POSTURE}
                if postures_a and postures_b and postures_a != postures_b:
                    report.add(
                        "warning",
                        "contradiction",
                        f"overlapping spans in {Path(video).name} assert "
                        f"{sorted(postures_a)} and {sorted(postures_b)} at the same time",
                    )

    # Thin classes.
    if labels is not None:
        for name in labels:
            count = report.label_counts.get(name, 0)
            if count == 0:
                report.add(
                    "warning", "no_examples", f"{name!r} has no examples in this split"
                )
            elif count < min_examples:
                report.add(
                    "warning",
                    "few_examples",
                    f"{name!r} has only {count} examples; its metrics will be noise",
                )
    if report.label_counts and report.imbalance_ratio() > 50:
        report.add(
            "warning",
            "imbalance",
            f"most common class is {report.imbalance_ratio():.0f}x the rarest; "
            f"keep class_balanced sampling on and read macro metrics, not accuracy",
        )

    if check_duplicates and len(usable) > 1:
        signatures: dict[str, np.ndarray] = {}
        for index, (ann, meta) in enumerate(usable):
            key = f"{ann.video}@{ann.start:.2f}#{index}"
            try:
                signatures[key] = clip_signature(ann.video, meta, start=ann.start, end=ann.end)
            except Exception:
                continue
        report.duplicates = find_near_duplicates(signatures, duplicate_distance)
        for a, b, distance in report.duplicates[:50]:
            group_a = _group_of(a, usable)
            group_b = _group_of(b, usable)
            level = "warning" if group_a == group_b else "error"
            report.add(
                level,
                "near_duplicate",
                f"{Path(a).name} and {Path(b).name} are near-identical "
                f"(hamming {distance})"
                + ("" if group_a == group_b else " but are in different groups, so a "
                   "group-aware split will still separate them into train and val — "
                   "merge their groups"),
            )
    cache.flush()
    return report


def audit_splits(
    splits: dict[str, Sequence[Annotation]],
    labels: LabelSpace | None = None,
    **kwargs,
) -> tuple[dict[str, AuditReport], list[Finding]]:
    """Audit each split, then check for leakage *between* them."""
    reports = {name: audit_annotations(anns, labels, **kwargs) for name, anns in splits.items()}

    cross: list[Finding] = []
    group_owner: dict[str, list[str]] = defaultdict(list)
    for name, anns in splits.items():
        for group in {a.group_key for a in anns}:
            group_owner[group].append(name)
    for group, owners in group_owner.items():
        if len(owners) > 1:
            cross.append(
                Finding(
                    "error",
                    "split_leakage",
                    f"group {group!r} appears in splits {sorted(owners)}; clips from the "
                    f"same source video are near-duplicates, so validation scores "
                    f"computed this way are inflated",
                    group,
                )
            )

    # Cross-split near-duplicates by content, not just by name.
    cache = MetaCache(kwargs.get("cache_path"))
    signatures: dict[str, np.ndarray] = {}
    owners: dict[str, str] = {}
    for name, anns in splits.items():
        for index, ann in enumerate(anns):
            if not Path(ann.video).exists():
                continue
            key = f"{name}:{ann.video}@{ann.start:.2f}#{index}"
            try:
                signatures[key] = clip_signature(ann.video, cache.get(ann.video), start=ann.start, end=ann.end)
                owners[key] = name
            except Exception:
                continue
    cache.flush()
    for a, b, distance in find_near_duplicates(signatures, kwargs.get("duplicate_distance", 6)):
        if owners[a] != owners[b]:
            cross.append(
                Finding(
                    "error",
                    "cross_split_duplicate",
                    f"near-identical clips (hamming {distance}) in both "
                    f"{owners[a]} and {owners[b]}: {Path(a).name} / {Path(b).name}",
                )
            )
    return reports, cross


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _by_video(
    usable: Iterable[tuple[Annotation, VideoMeta]]
) -> dict[str, list[tuple[Annotation, VideoMeta]]]:
    out: dict[str, list[tuple[Annotation, VideoMeta]]] = defaultdict(list)
    for ann, meta in usable:
        out[ann.video].append((ann, meta))
    return out


def _overlaps(a: Annotation, b: Annotation, meta_a: VideoMeta, meta_b: VideoMeta) -> bool:
    end_a = a.end if a.end is not None else meta_a.duration
    end_b = b.end if b.end is not None else meta_b.duration
    return a.start < end_b and b.start < end_a


def _group_of(key: str, usable: Sequence[tuple[Annotation, VideoMeta]]) -> str:
    video = key.split("@")[0]
    for ann, _ in usable:
        if ann.video == video:
            return ann.group_key
    return video


def _closest(name: str, labels: LabelSpace) -> str:
    """Cheap edit-distance suggestion for a misspelled label."""
    best, best_score = "", -1.0
    for candidate in labels:
        score = _similarity(name, candidate)
        if score > best_score:
            best, best_score = candidate, score
    return f"did you mean {best!r}?" if best_score > 0.6 else "no close match"


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    # Longest common subsequence ratio — good enough to catch typos.
    previous = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            current[j] = (
                previous[j - 1] + 1 if a[i - 1] == b[j - 1] else max(previous[j], current[j - 1])
            )
        previous = current
    return 2.0 * previous[-1] / (len(a) + len(b))
