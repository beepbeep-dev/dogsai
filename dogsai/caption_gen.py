"""Building a caption dataset for the narrator, from the signals in real footage.

The problem this solves
-----------------------
`DogNarrator` trained on the `dogbehaviour` captions reaches perplexity 1.04 and
100% exact match, and both numbers are hollow: the corpus contains **five distinct
caption strings**, so the task is a five-way lookup. There is nothing there to
learn about language.

This module builds a richer corpus over the *same real videos*. For each clip it
measures things the source captions ignore — how long the behaviour ran, whether
the dog vocalised and how, what kind of vocalisation, how aroused it sounded, how
much of the clip was vocal — and composes a sentence that states them. The result
is thousands of distinct captions whose wording varies with measurable properties
of the clip, so a model has to attend to the conditioning vector rather than
memorise five outputs.

What this is and is not
-----------------------
This is **not** human annotation, and it must not be described as such. The
captions are generated from detector output by templates, which has two concrete
consequences:

1. **It cannot teach the narrator anything the detectors do not already measure.**
   The ceiling is "fluent, compositional restatement of the feature vector". No
   amount of this data will teach it to describe something the pipeline cannot see,
   like where the dog is looking.
2. **It inherits every detector error.** If the audio module mistypes a squeaky toy
   as a whine, the caption says whine, and the narrator learns to say whine. Errors
   propagate rather than cancel.

So why do it? Because "verbalise the measurements, in varied and grammatical
English, conditioned on their values" is a real task with a real skill in it, and
it is exactly the job the narrator has at inference time. Learning it from data
generalises better than the fixed templates in :mod:`dogsai.translate` — the model
can interpolate between conditions those templates handle only by branching. The
honest framing is: this is a *distillation* corpus that teaches language, not a
knowledge corpus that teaches facts about dogs.

Determinism is by clip identity, so regenerating gives byte-identical captions and
the dataset is reproducible.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .audio import NOT_THE_DOG, AudioReading, read_audio
from .dataset import Annotation

# Behaviour -> the phrases that can describe it. Multiple surface forms per
# behaviour so wording varies independently of the label.
_ACTION_PHRASES: dict[str, tuple[str, ...]] = {
    "chewing": ("chewing on something", "working on a toy", "gnawing at something"),
    "eating_drinking": ("eating", "having a meal", "eating from a bowl"),
    "eliminating": ("relieving itself", "going to the toilet", "squatting"),
    "playing": ("playing", "playing with a toy", "messing about with a toy"),
    "yawning": ("yawning", "having a big yawn", "opening its mouth wide in a yawn"),
    "lying_down": ("lying down", "settled on the floor", "resting"),
    "sitting": ("sitting", "sat still", "sitting and waiting"),
    "standing": ("standing", "standing still", "on its feet"),
    "walking": ("walking", "wandering about", "moving at a walk"),
    "trotting": ("trotting", "moving briskly", "trotting along"),
    "running": ("running", "sprinting", "running flat out"),
    "jumping": ("jumping up", "leaping about", "bouncing up"),
    "sniffing": ("sniffing around", "nose to the ground", "investigating a smell"),
    "digging": ("digging", "scraping at the ground", "digging a hole"),
    "tail_wagging": ("wagging its tail", "tail going", "with its tail wagging"),
    "scratching": ("scratching itself", "having a scratch"),
    "shaking_off": ("shaking itself off", "having a shake"),
    "rolling": ("rolling around", "rolling on its back"),
    "stretching": ("stretching out", "having a stretch"),
    "alert_freeze": ("frozen still and staring", "locked on to something"),
    "barking": ("barking", "making noise"),
}

_VOICE_PHRASES: dict[str, tuple[str, ...]] = {
    "bark": ("barks", "lets out a bark", "gives a bark"),
    "bark_alarm": ("barks a warning", "barks low and hard", "sounds the alarm"),
    "bark_excited": ("barks excitedly", "yaps happily", "barks in a high, bright tone"),
    "growl": ("growls", "gives a low growl", "growls a warning"),
    "whine": ("whines", "lets out a whine", "whimpers"),
    "howl": ("howls", "lets out a long howl", "howls for company"),
    "yelp": ("yelps sharply", "gives a sudden yelp", "cries out"),
    "pant": ("pants heavily", "is panting", "breathes hard"),
}

_DURATION_WORDS: tuple[tuple[float, tuple[str, ...]], ...] = (
    (1.5, ("briefly", "for a moment", "just for a second")),
    (4.0, ("for a few seconds", "for a short while")),
    (8.0, ("for several seconds", "for a good while")),
    (1e9, ("for most of the clip", "throughout", "for the whole clip")),
)

_AROUSAL_WORDS: tuple[tuple[float, tuple[str, ...]], ...] = (
    (0.35, ("calmly", "quietly", "in a settled way")),
    (0.65, ("steadily", "without much fuss")),
    (1e9, ("energetically", "with a lot of energy", "excitedly")),
)

_COUNT_WORDS: dict[int, tuple[str, ...]] = {
    1: ("once",),
    2: ("twice",),
    3: ("three times", "a few times"),
}


def _pick(options: Sequence[str], rng: np.random.Generator) -> str:
    return str(options[int(rng.integers(0, len(options)))])


def _band(value: float, table: tuple[tuple[float, tuple[str, ...]], ...],
          rng: np.random.Generator) -> str:
    for threshold, options in table:
        if value < threshold:
            return _pick(options, rng)
    return _pick(table[-1][1], rng)


def _count_word(n: int, rng: np.random.Generator) -> str:
    if n in _COUNT_WORDS:
        return _pick(_COUNT_WORDS[n], rng)
    return f"{n} times"


@dataclass
class GeneratedCaption:
    """One caption plus the measurements it was composed from."""

    video: str
    caption: str
    labels: list[str]
    duration: float
    audio: dict
    group: str

    def to_annotation(self, start: float = 0.0, end: float | None = None) -> Annotation:
        return Annotation(
            video=self.video,
            labels=list(self.labels),
            start=start,
            end=end,
            group=self.group,
            meta={"caption": self.caption, "generated": True, "audio": self.audio},
        )


def compose_caption(
    labels: Sequence[str],
    duration: float,
    audio: AudioReading | None,
    seed: int,
) -> str:
    """Compose one sentence from a clip's measured properties.

    Sentence structure varies (behaviour-first, voice-first, or both clauses) and
    so does word choice, both keyed on ``seed`` — derived from clip identity by the
    caller, so the corpus is reproducible.
    """
    rng = np.random.default_rng(seed)
    behaviour = labels[0] if labels else None
    phrases = _ACTION_PHRASES.get(behaviour or "", ("doing something",))
    action = _pick(phrases, rng)
    when = _band(duration, _DURATION_WORDS, rng)

    voice_events = [e for e in (audio.events if audio else []) if e.kind not in NOT_THE_DOG]
    counts: dict[str, int] = {}
    for event in voice_events:
        counts[event.kind] = counts.get(event.kind, 0) + 1
    loudest = max(voice_events, key=lambda e: e.confidence) if voice_events else None

    clauses: list[str] = []
    subject = _pick(("The dog", "The dog", "It"), rng)

    if behaviour:
        manner = ""
        if audio is not None and audio.confidence > 0.2 and rng.random() < 0.5:
            manner = " " + _band(audio.arousal, _AROUSAL_WORDS, rng)
        clauses.append(f"{subject} is {action}{manner} {when}")

    if loudest is not None:
        kind = loudest.kind
        phrase = _pick(_VOICE_PHRASES.get(kind, ("makes a noise",)), rng)
        n = counts.get(kind, 1)
        times = "" if n == 1 and rng.random() < 0.5 else " " + _count_word(n, rng)
        other = [k for k in counts if k != kind]
        if clauses:
            connector = _pick((" and ", ", and ", " while it "), rng)
            if connector == " while it ":
                clauses.append(f"{connector}{phrase}{times}")
            else:
                clauses.append(f"{connector}it {phrase}{times}")
        else:
            clauses.append(f"{subject} {phrase}{times}")
        if other and rng.random() < 0.55:
            extra = _pick(_VOICE_PHRASES.get(other[0], ("makes a noise",)), rng)
            clauses.append(f", then {extra}")
    elif clauses and rng.random() < 0.45:
        clauses.append(_pick((" and stays quiet", " without making a sound",
                              " and is silent"), rng))

    if not clauses:
        return "Nothing much happens in this clip."
    sentence = "".join(clauses).strip()
    sentence = sentence[0].upper() + sentence[1:]
    return sentence.rstrip(".,") + "."


def generate_captions(
    annotations: Sequence[Annotation],
    with_audio: bool = True,
    verbose: bool = True,
) -> list[GeneratedCaption]:
    """Measure every clip and compose a caption for it."""
    out: list[GeneratedCaption] = []
    for i, annotation in enumerate(annotations):
        duration = (annotation.end or 0.0) - annotation.start
        if duration <= 0:
            duration = 0.0
        reading: AudioReading | None = None
        if with_audio:
            try:
                reading = read_audio(annotation.video)
            except Exception:
                reading = None
        # Seed from the clip's identity, so regenerating is byte-identical.
        digest = hashlib.sha256(
            f"{annotation.video}:{annotation.start:.3f}".encode()
        ).digest()
        seed = int.from_bytes(digest[:8], "big")
        caption = compose_caption(annotation.labels, duration, reading, seed)
        out.append(
            GeneratedCaption(
                video=annotation.video,
                caption=caption,
                labels=list(annotation.labels),
                duration=duration,
                audio=reading.to_dict() if reading else {},
                group=annotation.group_key,
            )
        )
        if verbose and (i + 1) % 200 == 0:
            print(f"  captioned {i + 1}/{len(annotations)}", flush=True)
    return out


def corpus_stats(captions: Sequence[GeneratedCaption]) -> dict:
    """Diversity statistics — the numbers that say whether this was worth doing."""
    texts = [c.caption for c in captions]
    words = [w for t in texts for w in t.lower().split()]
    lengths = [len(t.split()) for t in texts]
    return {
        "captions": len(texts),
        "distinct": len(set(texts)),
        "distinct_ratio": len(set(texts)) / max(1, len(texts)),
        "vocabulary": len(set(words)),
        "mean_words": float(np.mean(lengths)) if lengths else 0.0,
        "min_words": int(np.min(lengths)) if lengths else 0,
        "max_words": int(np.max(lengths)) if lengths else 0,
    }


def save_corpus(path: str | Path, captions: Sequence[GeneratedCaption]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for item in captions:
            handle.write(json.dumps({
                "video": item.video,
                "caption": item.caption,
                "labels": item.labels,
                "duration": round(item.duration, 3),
                "group": item.group,
                "audio": {
                    "kinds": item.audio.get("kinds", {}),
                    "arousal": item.audio.get("arousal", 0.0),
                    "valence": item.audio.get("valence", 0.0),
                    "vocal_fraction": item.audio.get("vocal_fraction", 0.0),
                },
            }) + "\n")
    return path


def load_corpus(path: str | Path) -> list[GeneratedCaption]:
    out: list[GeneratedCaption] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        out.append(GeneratedCaption(
            video=record["video"],
            caption=record["caption"],
            labels=record.get("labels", []),
            duration=float(record.get("duration", 0.0)),
            audio=record.get("audio", {}),
            group=record.get("group", ""),
        ))
    return out


def corpus_to_examples(captions: Sequence[GeneratedCaption], behaviours: list[str]):
    """Turn a corpus into narrator training examples."""
    from .narrate import NarrationExample, condition_vector

    index = {name: i for i, name in enumerate(behaviours)}
    examples = []
    for item in captions:
        scores = np.zeros(len(behaviours), dtype=np.float32)
        for label in item.labels:
            if label in index:
                scores[index[label]] = 1.0
        if scores.sum() > 0:
            scores = scores * 0.85 + 0.15 / len(behaviours)
        examples.append(NarrationExample(
            condition=condition_vector(scores, item.audio, item.duration),
            caption=item.caption,
        ))
    return examples
