"""Building a richer training dataset for the main model, using our own tools.

What "our own dataset" means here
----------------------------------
The `dogbehaviour` source gives each clip exactly one behaviour label. That is a
real ceiling: a clip captioned "Dog is eating." that also barks partway through is
trained as pure "eating_drinking", and the model never gets to learn what a bark
looks like in that clip because nothing says one is there.

This module removes part of that ceiling using a detector we already built and
verified, rather than any new external annotation. :mod:`dogsai.audio` listens to
every clip and types its vocalisations; wherever it hears a confident, sustained
bark, that clip earns an additional ``"barking"`` label alongside its original one.
The result is a genuinely multi-label dataset built from nothing but the video
files already on disk and code in this repository — no scraping, no purchased
annotation, no human labeller.

Two honest limits, stated plainly:

1. **This adds recall for one behaviour, not ground truth for all of them.** It
   only ever *adds* the "barking" label; it never removes or second-guesses the
   original human-authored label. So it is a strict enrichment, not a re-annotation.
2. **It inherits the audio detector's error rate.** `dogsai/audio.py` types
   vocalisations by rule-based DSP with real but imperfect precision (see its
   module docstring), and a clip mistyped as containing a bark becomes a false
   "barking" label here. The ``min_count``/``min_confidence`` thresholds trade
   recall for precision; the defaults favour precision, because false positives
   corrupt training while false negatives just forgo a little signal.

Re-running produces byte-identical output for a fixed detector and thresholds,
because it is a pure function of the audio and the existing labels — nothing here
is sampled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .audio import read_audio
from .dataset import Annotation, load_annotations, save_annotations

# The only audio-detected category that also names a class in the video
# taxonomy (dogsai.labels.ACTIONS). Growls, whines, howls and yelps are real
# vocalisation types but have no corresponding *visual* behaviour class to
# enrich onto — adding them here would invent labels the taxonomy has no
# meaning for.
BARK_KINDS: tuple[str, ...] = ("bark", "bark_alarm", "bark_excited")
ADDED_LABEL = "barking"


@dataclass
class EnrichmentStats:
    total: int = 0
    already_labelled: int = 0
    added: int = 0
    no_audio_signal: int = 0
    skipped_missing_audio: int = 0

    def render(self) -> str:
        return (
            f"{self.total} clips: {self.added} gained '{ADDED_LABEL}', "
            f"{self.already_labelled} already had it, "
            f"{self.no_audio_signal} had no qualifying bark, "
            f"{self.skipped_missing_audio} had no audio data available"
        )


def _bark_evidence(audio_summary: dict | None, bark_kinds: Sequence[str]) -> tuple[int, float]:
    """Total bark-type event count and the clip's audio confidence."""
    if not audio_summary:
        return 0, 0.0
    kinds = audio_summary.get("kinds", {}) or {}
    count = sum(int(kinds.get(k, 0)) for k in bark_kinds)
    return count, float(audio_summary.get("confidence", 0.0) or 0.0)


def enrich_annotations(
    annotations: Sequence[Annotation],
    audio_index: dict[str, dict] | None = None,
    bark_kinds: Sequence[str] = BARK_KINDS,
    min_count: int = 1,
    min_confidence: float = 0.0,
    compute_missing: bool = True,
    verbose: bool = False,
) -> tuple[list[Annotation], EnrichmentStats]:
    """Add ``"barking"`` to any clip where our audio detector heard one confidently.

    ``audio_index`` maps video path -> an audio summary dict (as produced by
    :meth:`dogsai.audio.AudioReading.to_dict`), letting the caller reuse audio
    already computed elsewhere (e.g. by ``dogsai make-captions``) instead of
    re-decoding every clip. Missing entries fall back to computing it live when
    ``compute_missing`` is set, or are left unenriched otherwise.
    """
    audio_index = audio_index or {}
    stats = EnrichmentStats(total=len(annotations))
    out: list[Annotation] = []

    for annotation in annotations:
        if ADDED_LABEL in annotation.labels:
            stats.already_labelled += 1
            out.append(annotation)
            continue

        summary = audio_index.get(annotation.video)
        if summary is None and compute_missing:
            try:
                summary = read_audio(annotation.video).to_dict()
            except Exception:
                summary = None
        if summary is None:
            stats.skipped_missing_audio += 1
            out.append(annotation)
            continue

        count, confidence = _bark_evidence(summary, bark_kinds)
        if count >= min_count and confidence >= min_confidence:
            enriched = Annotation(
                video=annotation.video,
                labels=[*annotation.labels, ADDED_LABEL],
                start=annotation.start,
                end=annotation.end,
                group=annotation.group,
                subject=annotation.subject,
                weight=annotation.weight,
                meta={**annotation.meta, "audio_enriched": True, "bark_events": count},
            )
            out.append(enriched)
            stats.added += 1
            if verbose:
                print(f"  + barking: {Path(annotation.video).name} ({count} bark event(s))")
        else:
            stats.no_audio_signal += 1
            out.append(annotation)

    return out, stats


def load_audio_index(captions_paths: Sequence[str | Path]) -> dict[str, dict]:
    """Build a video -> audio-summary lookup from ``dogsai make-captions`` output."""
    index: dict[str, dict] = {}
    for path in captions_paths:
        path = Path(path)
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            index[record["video"]] = record.get("audio", {})
    return index


def enrich_dataset(
    data_root: str | Path,
    out_root: str | Path,
    splits: Sequence[str] = ("train", "val"),
    captions_dir: str | Path | None = None,
    bark_kinds: Sequence[str] = BARK_KINDS,
    min_count: int = 1,
    min_confidence: float = 0.0,
    verbose: bool = True,
) -> dict[str, EnrichmentStats]:
    """Enrich every split of a prepared dataset and write it under ``out_root``.

    Also (re)writes ``behaviours.txt`` with ``"barking"`` appended if it is not
    already present, since training on the enriched data needs it in the label
    space.
    """
    data_root = Path(data_root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    captions_dir = Path(captions_dir) if captions_dir else data_root / "captions"
    audio_index = load_audio_index(
        [captions_dir / f"{split}_captions.jsonl" for split in splits]
    )
    if verbose:
        print(f"audio index: {len(audio_index)} clips "
              f"({'from ' + str(captions_dir) if audio_index else 'none found, will compute live'})")

    all_stats: dict[str, EnrichmentStats] = {}
    behaviours: set[str] = set()
    for split in splits:
        source = data_root / f"{split}.jsonl"
        if not source.exists():
            continue
        annotations = load_annotations(source)
        enriched, stats = enrich_annotations(
            annotations, audio_index, bark_kinds, min_count, min_confidence,
            verbose=False,  # per-clip logging is only useful for debugging
        )
        save_annotations(out_root / f"{split}.jsonl", enriched)
        all_stats[split] = stats
        for annotation in enriched:
            behaviours.update(annotation.labels)
        if verbose:
            print(f"{split}: {stats.render()}")

    # The label space must cover the full original taxonomy, not just whatever
    # happens to appear in this particular split — a class with zero examples in
    # `train` (e.g. it only occurs in `val`, or a class list narrower than the
    # source split by construction) must not silently disappear from the model's
    # output space. So this always unions the pre-existing behaviours.txt (if any)
    # with whatever labels actually appeared, plus the label this function adds.
    existing = data_root / "behaviours.txt"
    existing_names = (
        {n.strip() for n in existing.read_text().split() if n.strip()}
        if existing.exists()
        else set()
    )
    names = sorted(existing_names | behaviours | {ADDED_LABEL})
    (out_root / "behaviours.txt").write_text("\n".join(names) + "\n")
    return all_stats
