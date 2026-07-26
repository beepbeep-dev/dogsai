"""Importers that turn real annotation exports into ``Annotation`` records.

The point of this module is to make it cheap to bring *your* footage in, because
the fastest route to an accurate model is more real, correctly-labelled video —
not more architecture.  Each importer normalises into the same span-based form
and, crucially, sets a sensible ``group`` so the splitter cannot leak.

Supported sources:

* ``label-studio`` — Label Studio JSON export with ``VideoTimeline`` / ``Labels``
  regions (the most common hand-annotation route for this task).
* ``cvat`` — CVAT for Video XML export, tag-based annotation.
* ``csv`` — any CSV with ``video,start,end,labels`` columns; the escape hatch.
* ``ava`` — AVA-style CSV (``video_id,timestamp,...,action_id``), used by several
  animal-behaviour datasets including the Animal Kingdom action split.
* ``folders`` — ``root/<behaviour>/clip.mp4``.

Label names are passed through an optional mapping so an external taxonomy can be
folded into this one (``{"run": "running", "trot": "trotting"}``) — and anything
unmapped is reported rather than silently dropped.
"""

from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

from .dataset import Annotation, annotations_from_folders
from .video import VIDEO_SUFFIXES


class ImportResult:
    """Annotations plus a record of what could not be mapped."""

    def __init__(self, annotations: list[Annotation], unmapped: Counter | None = None):
        self.annotations = annotations
        self.unmapped = unmapped or Counter()

    def __len__(self) -> int:
        return len(self.annotations)

    def report(self) -> str:
        labels = Counter(name for a in self.annotations for name in a.labels)
        lines = [
            f"imported {len(self.annotations)} spans "
            f"across {len({a.video for a in self.annotations})} videos, "
            f"{len({a.group_key for a in self.annotations})} groups",
        ]
        for name, count in labels.most_common():
            lines.append(f"  {name:<20} {count}")
        if self.unmapped:
            lines.append("unmapped source labels (add them to --label-map to keep them):")
            for name, count in self.unmapped.most_common():
                lines.append(f"  {name:<20} {count}  DROPPED")
        return "\n".join(lines)


def _apply_map(
    names: Iterable[str],
    mapping: dict[str, str] | None,
    unmapped: Counter,
    keep_unmapped: bool,
) -> list[str]:
    out: list[str] = []
    for raw in names:
        name = raw.strip()
        if not name:
            continue
        if mapping is None:
            out.append(name)
            continue
        key = name.lower().replace(" ", "_").replace("-", "_")
        if key in mapping:
            out.append(mapping[key])
        elif name in mapping:
            out.append(mapping[name])
        elif keep_unmapped:
            out.append(key)
        else:
            unmapped[name] += 1
    # De-duplicate while preserving order (a mapping can collapse two source
    # labels onto one target).
    seen: set[str] = set()
    return [n for n in out if not (n in seen or seen.add(n))]


def _resolve_video(name: str, video_root: Path | None) -> str:
    """Match an annotation's video reference to a file on disk.

    Exports routinely store a URL, an upload path with a hash prefix, or a bare
    id with no extension, so a plain join is not enough.
    """
    if video_root is None:
        return name
    candidate = Path(name)
    direct = video_root / candidate.name
    if direct.exists():
        return str(direct)
    stem = candidate.stem
    # Label Studio prefixes uploads with an 8-char hash: "a1b2c3d4-clip.mp4".
    if "-" in stem:
        trimmed = stem.split("-", 1)[1]
        for suffix in VIDEO_SUFFIXES:
            probe = video_root / f"{trimmed}{suffix}"
            if probe.exists():
                return str(probe)
    for suffix in VIDEO_SUFFIXES:
        probe = video_root / f"{stem}{suffix}"
        if probe.exists():
            return str(probe)
    matches = sorted(video_root.rglob(f"{stem}.*"))
    matches = [m for m in matches if m.suffix.lower() in VIDEO_SUFFIXES]
    return str(matches[0]) if matches else name


# ---------------------------------------------------------------------------
# Label Studio
# ---------------------------------------------------------------------------
def from_label_studio(
    path: str | Path,
    video_root: str | Path | None = None,
    label_map: dict[str, str] | None = None,
    keep_unmapped: bool = False,
    default_fps: float = 30.0,
) -> ImportResult:
    """Parse a Label Studio JSON export.

    Handles both ``videorectangle``/``timelinelabels`` (frame-indexed) and
    ``labels`` on a video (second-indexed) result types.
    """
    root = Path(video_root) if video_root else None
    blob = json.loads(Path(path).read_text())
    tasks = blob if isinstance(blob, list) else blob.get("tasks", [])
    unmapped: Counter = Counter()
    out: list[Annotation] = []

    for task in tasks:
        data = task.get("data", {})
        source = next(
            (v for k, v in data.items() if isinstance(v, str) and ("video" in k or Path(v).suffix.lower() in VIDEO_SUFFIXES)),
            None,
        )
        if not source:
            continue
        video = _resolve_video(source, root)
        group = Path(source).stem
        annotations = task.get("annotations") or task.get("completions") or []
        for record in annotations:
            if record.get("was_cancelled"):
                continue
            for region in record.get("result", []):
                value = region.get("value", {})
                names = (
                    value.get("timelinelabels")
                    or value.get("labels")
                    or value.get("choices")
                    or []
                )
                labels = _apply_map(names, label_map, unmapped, keep_unmapped)
                if not labels and names:
                    continue

                fps = float(value.get("framesCount") and task.get("fps") or default_fps)
                start = end = None
                if "ranges" in value and value["ranges"]:
                    for span in value["ranges"]:
                        start = float(span.get("start", 0)) / fps
                        end = float(span.get("end", 0)) / fps
                        out.append(
                            Annotation(video, labels, start, end, group=group,
                                       meta={"source": "label-studio"})
                        )
                    continue
                if "start" in value:
                    start = float(value["start"])
                    end = float(value.get("end", start))
                    out.append(
                        Annotation(video, labels, start, end, group=group,
                                   meta={"source": "label-studio"})
                    )
                elif labels:
                    out.append(
                        Annotation(video, labels, 0.0, None, group=group,
                                   meta={"source": "label-studio"})
                    )
    return ImportResult(out, unmapped)


# ---------------------------------------------------------------------------
# CVAT
# ---------------------------------------------------------------------------
def from_cvat(
    path: str | Path,
    video_root: str | Path | None = None,
    label_map: dict[str, str] | None = None,
    keep_unmapped: bool = False,
    default_fps: float = 30.0,
) -> ImportResult:
    """Parse a CVAT-for-Video XML export (tags and track-level labels)."""
    root_dir = Path(video_root) if video_root else None
    tree = ET.parse(Path(path))
    xml_root = tree.getroot()
    unmapped: Counter = Counter()

    source_name = xml_root.findtext(".//source") or xml_root.findtext(".//name") or "video"
    fps_text = xml_root.findtext(".//original_size/../fps") or xml_root.findtext(".//fps")
    fps = float(fps_text) if fps_text else default_fps
    video = _resolve_video(source_name, root_dir)
    group = Path(source_name).stem

    # Frame-indexed tags: collapse consecutive frames carrying the same label.
    per_label: dict[str, list[int]] = {}
    for tag in xml_root.iter("tag"):
        raw = tag.get("label")
        frame = tag.get("frame")
        if raw is None or frame is None:
            continue
        labels = _apply_map([raw], label_map, unmapped, keep_unmapped)
        for name in labels:
            per_label.setdefault(name, []).append(int(frame))

    out: list[Annotation] = []
    for name, frames in per_label.items():
        frames.sort()
        run_start = previous = frames[0]
        for frame in frames[1:] + [None]:  # type: ignore[list-item]
            if frame is not None and frame - previous <= 1:
                previous = frame
                continue
            out.append(
                Annotation(video, [name], run_start / fps, (previous + 1) / fps,
                           group=group, meta={"source": "cvat"})
            )
            if frame is not None:
                run_start = previous = frame

    for track in xml_root.iter("track"):
        raw = track.get("label")
        if raw is None:
            continue
        labels = _apply_map([raw], label_map, unmapped, keep_unmapped)
        frames = [int(box.get("frame", 0)) for box in track if box.get("outside") != "1"]
        if labels and frames:
            out.append(
                Annotation(video, labels, min(frames) / fps, (max(frames) + 1) / fps,
                           group=group, meta={"source": "cvat"})
            )
    return ImportResult(out, unmapped)


# ---------------------------------------------------------------------------
# CSV variants
# ---------------------------------------------------------------------------
def from_csv(
    path: str | Path,
    video_root: str | Path | None = None,
    label_map: dict[str, str] | None = None,
    keep_unmapped: bool = True,
    label_separator: str = "|",
) -> ImportResult:
    """Parse ``video,start,end,labels[,group,subject]`` CSV.

    Column names are matched case-insensitively and a few aliases are accepted
    (``file``/``path``/``clip`` for ``video``, ``label``/``behaviour`` for
    ``labels``) because every lab names them differently.
    """
    root = Path(video_root) if video_root else None
    unmapped: Counter = Counter()
    out: list[Annotation] = []
    aliases = {
        "video": {"video", "file", "path", "clip", "filename", "video_id"},
        "start": {"start", "start_time", "begin", "from", "t0"},
        "end": {"end", "end_time", "stop", "to", "t1"},
        "labels": {"labels", "label", "behaviour", "behavior", "action", "class"},
        "group": {"group", "source", "session", "video_group"},
        "subject": {"subject", "dog", "animal", "individual"},
    }

    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no header row")
        lookup: dict[str, str] = {}
        for column in reader.fieldnames:
            key = column.strip().lower()
            for canonical, names in aliases.items():
                if key in names:
                    lookup[canonical] = column
        if "video" not in lookup or "labels" not in lookup:
            raise ValueError(
                f"{path}: need a video column and a labels column; got {reader.fieldnames}"
            )
        for row in reader:
            raw_video = (row.get(lookup["video"]) or "").strip()
            if not raw_video:
                continue
            names = (row.get(lookup["labels"]) or "").split(label_separator)
            labels = _apply_map(names, label_map, unmapped, keep_unmapped)
            start = float(row.get(lookup.get("start", ""), 0) or 0)
            end_raw = row.get(lookup.get("end", ""), "") or ""
            video = _resolve_video(raw_video, root)
            out.append(
                Annotation(
                    video=video,
                    labels=labels,
                    start=start,
                    end=float(end_raw) if end_raw.strip() else None,
                    group=(row.get(lookup.get("group", "")) or None) or Path(raw_video).stem,
                    subject=row.get(lookup.get("subject", "")) or None,
                    meta={"source": "csv"},
                )
            )
    return ImportResult(out, unmapped)


def from_ava_csv(
    path: str | Path,
    video_root: str | Path | None = None,
    label_map: dict[str, str] | None = None,
    action_names: dict[int, str] | None = None,
    keep_unmapped: bool = False,
    window: float = 1.5,
) -> ImportResult:
    """Parse AVA-style ``video_id,timestamp,x1,y1,x2,y2,action_id,...`` rows.

    AVA labels a single keyframe timestamp, so each row becomes a span centred on
    that timestamp with a ``window``-second extent.  Rows for the same video,
    timestamp and different actions are merged into one multi-label span, which is
    what the ``multilabel`` head expects.
    """
    root = Path(video_root) if video_root else None
    unmapped: Counter = Counter()
    merged: dict[tuple[str, float], set[str]] = {}

    with Path(path).open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 2 or row[0].startswith("#"):
                continue
            video_id, timestamp = row[0], float(row[1])
            action = row[6] if len(row) > 6 else (row[2] if len(row) > 2 else "")
            raw = action.strip()
            if action_names and raw.isdigit():
                raw = action_names.get(int(raw), raw)
            labels = _apply_map([raw], label_map, unmapped, keep_unmapped)
            if labels:
                merged.setdefault((video_id, timestamp), set()).update(labels)

    out: list[Annotation] = []
    for (video_id, timestamp), labels in sorted(merged.items()):
        out.append(
            Annotation(
                video=_resolve_video(video_id, root),
                labels=sorted(labels),
                start=max(0.0, timestamp - window / 2),
                end=timestamp + window / 2,
                group=Path(video_id).stem,
                meta={"source": "ava", "keyframe": timestamp},
            )
        )
    return ImportResult(out, unmapped)


def from_folders(
    root: str | Path,
    label_map: dict[str, str] | None = None,
    keep_unmapped: bool = True,
) -> ImportResult:
    """Wrap folder-mode discovery, applying the label map."""
    unmapped: Counter = Counter()
    out: list[Annotation] = []
    for ann in annotations_from_folders(root):
        labels = _apply_map(ann.labels, label_map, unmapped, keep_unmapped)
        if labels:
            ann.labels = labels
            out.append(ann)
    return ImportResult(out, unmapped)


IMPORTERS: dict[str, Callable[..., ImportResult]] = {
    "label-studio": from_label_studio,
    "cvat": from_cvat,
    "csv": from_csv,
    "ava": from_ava_csv,
    "folders": from_folders,
}


def load_label_map(path: str | Path | None) -> dict[str, str] | None:
    """Read a ``{"source_label": "target_behaviour"}`` JSON mapping."""
    if not path:
        return None
    raw = json.loads(Path(path).read_text())
    return {str(k).lower().replace(" ", "_").replace("-", "_"): str(v) for k, v in raw.items()}
