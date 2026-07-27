"""Fetching and converting real, public dog-behaviour video datasets.

The model in this package is written from scratch, but it still has to learn from
real footage — a network trained on rendered figures recognises rendered figures.
This module handles the acquisition side: download a public dataset, convert its
annotations into :class:`~dogsai.dataset.Annotation` spans, map its label
vocabulary onto this taxonomy, and split it group-aware.

Registered datasets are described declaratively in :data:`REGISTRY` so adding one
is a data change, not a code change. Each entry records its licence, because
these are other people's datasets and the terms travel with them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .dataset import Annotation, make_splits, save_annotations


@dataclass(frozen=True)
class DatasetSpec:
    """A downloadable dataset and how to read it."""

    name: str
    repo_id: str
    kind: str  # huggingface repo_type
    description: str
    licence: str
    approx_gb: float
    behaviours: tuple[str, ...]
    label_map: dict[str, str] = field(default_factory=dict)
    allow_patterns: tuple[str, ...] | None = None
    notes: str = ""

    def summary(self) -> str:
        return (
            f"{self.name}\n"
            f"  source    : {self.repo_id} (HuggingFace {self.kind})\n"
            f"  size      : ~{self.approx_gb:.1f} GB\n"
            f"  licence   : {self.licence}\n"
            f"  behaviours: {', '.join(self.behaviours)}\n"
            f"  {self.description}"
            + (f"\n  note: {self.notes}" if self.notes else "")
        )


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
REGISTRY: dict[str, DatasetSpec] = {
    "dogbehaviour": DatasetSpec(
        name="dog-behaviour (fishchen)",
        repo_id="fishchen/dog-behavior-dataset",
        kind="dataset",
        description=(
            "1217 short real-world dog clips, one behaviour label each, with a "
            "caption span per clip. Roughly balanced across five behaviours."
        ),
        licence="see the dataset card on HuggingFace before redistributing",
        approx_gb=9.5,
        behaviours=("yawning", "eating_drinking", "eliminating", "playing", "chewing"),
        label_map={
            # source vocabulary -> this taxonomy
            "yawn": "yawning",
            "eating": "eating_drinking",
            "pooping": "eliminating",
            "toy": "playing",
            "rope": "chewing",
        },
        notes=(
            "Labels are whole-clip, so a clip labelled 'yawning' contains a yawn "
            "somewhere in it rather than throughout. The caption span is used as "
            "the annotated interval, which is tighter than the full clip but still "
            "coarser than a hand-marked event."
        ),
    ),
    "animalkingdom": DatasetSpec(
        name="Animal Kingdom (action recognition split)",
        repo_id="Hzzone/Animal_Kingdom",
        kind="dataset",
        description=(
            "140 action classes across 850 species, video. Multi-species: needs "
            "filtering to dogs/canids, and its action vocabulary is broader than "
            "this taxonomy."
        ),
        licence="research use; see the original Animal Kingdom terms (CVPR 2022)",
        approx_gb=15.1,
        behaviours=("varies — 140 action classes",),
        notes="A single 15 GB tarball; expect to unpack and filter before use.",
    ),
    "mammalnet": DatasetSpec(
        name="MammalNet",
        repo_id="linxxx3/MammalNet",
        kind="dataset",
        description=(
            "18k videos, 173 mammal categories, 12 common behaviours. Large and "
            "taxonomically broad; the canid subset is a good source of real "
            "locomotion and feeding footage."
        ),
        licence="research use; see the original MammalNet terms (CVPR 2023)",
        approx_gb=159.0,
        behaviours=("12 behaviours incl. eating, running, sleeping, grooming"),
        notes="159 GB as one tarball — needs a machine with real disk.",
    ),
}


def describe_registry() -> str:
    return "\n\n".join(spec.summary() for spec in REGISTRY.values())


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------
def download(
    key: str,
    root: str | Path,
    workers: int = 8,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Fetch a registered dataset into ``root``.

    Resumable: ``snapshot_download`` skips files already present, so an
    interrupted download continues rather than restarting.
    """
    if key not in REGISTRY:
        raise KeyError(f"unknown dataset {key!r}; known: {', '.join(REGISTRY)}")
    spec = REGISTRY[key]
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "downloading needs huggingface_hub: pip install huggingface_hub"
        ) from exc

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    patterns = allow_patterns or (list(spec.allow_patterns) if spec.allow_patterns else None)
    path = snapshot_download(
        spec.repo_id,
        repo_type=spec.kind,
        local_dir=str(root),
        max_workers=workers,
        allow_patterns=patterns,
    )
    return Path(path)


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------
def convert_dogbehaviour(
    raw_root: str | Path,
    label_map: dict[str, str] | None = None,
    pad: float = 0.0,
) -> list[Annotation]:
    """Convert ``fishchen/dog-behavior-dataset`` metadata into span annotations.

    The source gives one behaviour and one caption interval per clip. The caption
    interval becomes the annotated span, which keeps the sampler inside the part
    of the clip the annotator was describing.

    Every clip is its own group. That is the honest choice here: the clips are
    independently sourced, so filename-level grouping is the finest correct
    granularity, and nothing stronger can be inferred from the metadata.
    """
    raw_root = Path(raw_root)
    metadata = raw_root / "data" / "metadata.jsonl"
    if not metadata.exists():
        raise FileNotFoundError(f"expected metadata at {metadata}")
    mapping = label_map if label_map is not None else REGISTRY["dogbehaviour"].label_map

    out: list[Annotation] = []
    for line in metadata.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        video = raw_root / record["video"]
        if not video.exists():
            continue
        raw_labels = record.get("behaviors") or []
        labels = [mapping.get(str(b).lower(), str(b).lower()) for b in raw_labels]
        labels = [l for l in dict.fromkeys(labels) if l]
        if not labels:
            continue
        captions = record.get("captions") or []
        if captions:
            start = min(float(c.get("start", 0.0)) for c in captions)
            end = max(float(c.get("end", 0.0)) for c in captions)
        else:
            start, end = 0.0, None
        if end is not None:
            start = max(0.0, start - pad)
            end = end + pad
            if end <= start:
                end = None
        out.append(
            Annotation(
                video=str(video),
                labels=labels,
                start=start,
                end=end,
                group=video.stem,
                meta={"source": "fishchen/dog-behavior-dataset",
                      "raw_labels": raw_labels,
                      # Kept for dogsai.narrate: the free-text caption is the
                      # only natural-language supervision this dataset carries.
                      "caption": " ".join(
                          str(c.get("text", "")).strip() for c in captions
                      ).strip()},
            )
        )
    return out


CONVERTERS: dict[str, Callable[..., list[Annotation]]] = {
    "dogbehaviour": convert_dogbehaviour,
}


def prepare(
    key: str,
    raw_root: str | Path,
    out_root: str | Path,
    fractions: dict[str, float] | None = None,
    seed: int = 0,
    relative: bool = True,
) -> dict[str, Path]:
    """Convert a downloaded dataset into ``train.jsonl`` / ``val.jsonl`` splits."""
    if key not in CONVERTERS:
        raise KeyError(
            f"no converter for {key!r}; available: {', '.join(CONVERTERS)}"
        )
    annotations = CONVERTERS[key](raw_root)
    if not annotations:
        raise ValueError(f"converted zero annotations from {raw_root}")

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    parts = make_splits(annotations, fractions or {"train": 0.8, "val": 0.2}, seed=seed)

    written: dict[str, Path] = {}
    for name, subset in parts.items():
        if relative:
            for annotation in subset:
                try:
                    annotation.video = str(
                        Path(annotation.video).resolve().relative_to(out_root.resolve())
                    )
                except ValueError:
                    pass  # dataset lives outside out_root; keep the absolute path
        written[name] = save_annotations(out_root / f"{name}.jsonl", subset)

    names = sorted({n for a in annotations for n in a.labels})
    (out_root / "behaviours.txt").write_text("\n".join(names) + "\n")
    written["behaviours"] = out_root / "behaviours.txt"
    return written
