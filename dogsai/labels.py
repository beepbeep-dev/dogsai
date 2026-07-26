"""Dog behaviour taxonomy.

The default taxonomy is deliberately split into two axes that a video model can
actually separate:

* **posture / locomotion** — what the whole body is doing (mutually exclusive in
  practice: a dog is not sitting and running at the same time).
* **action** — what the dog is doing on top of that posture (a running dog can
  also be barking, a sitting dog can wag its tail).

That split is why the pipeline supports both a single-label head (``multiclass``)
and a per-behaviour head (``multilabel``).  Real footage is overwhelmingly
multi-label, so ``multilabel`` is the default for the full taxonomy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

# --- posture / locomotion ------------------------------------------------
POSTURE: tuple[str, ...] = (
    "lying_down",
    "sitting",
    "standing",
    "walking",
    "trotting",
    "running",
)

# --- actions ------------------------------------------------------------
ACTIONS: tuple[str, ...] = (
    "jumping",
    "playing",
    "chewing",
    "eating_drinking",
    "sniffing",
    "digging",
    "barking",
    "tail_wagging",
    "scratching",
    "shaking_off",
    "rolling",
    "stretching",
    "yawning",
    "eliminating",
    "alert_freeze",
)

DEFAULT_BEHAVIOURS: tuple[str, ...] = POSTURE + ACTIONS

# Behaviours that are commonly read as stress / appeasement signals in dogs —
# "displacement behaviours" in the ethology literature.  Not a diagnosis: a
# yawn is also just a yawn.  This is a flag for "a human may want to watch
# this clip", nothing more.  See dogsai.affect for how these are used.
AROUSAL_FLAGS: frozenset[str] = frozenset(
    {"barking", "alert_freeze", "scratching", "shaking_off", "yawning"}
)


@dataclass(frozen=True)
class LabelSpace:
    """An ordered set of behaviour names plus the index bookkeeping around it."""

    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.names:
            raise ValueError("label space must contain at least one behaviour")
        dupes = {n for n in self.names if self.names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate behaviour names: {sorted(dupes)}")

    @classmethod
    def default(cls) -> "LabelSpace":
        return cls(DEFAULT_BEHAVIOURS)

    @classmethod
    def from_names(cls, names: Sequence[str]) -> "LabelSpace":
        return cls(tuple(names))

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self):
        return iter(self.names)

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:  # pragma: no cover - message clarity only
            raise KeyError(
                f"unknown behaviour {name!r}; known: {', '.join(self.names)}"
            ) from exc

    def name(self, index: int) -> str:
        return self.names[index]

    @property
    def posture_indices(self) -> tuple[int, ...]:
        return tuple(i for i, n in enumerate(self.names) if n in POSTURE)

    @property
    def action_indices(self) -> tuple[int, ...]:
        return tuple(i for i, n in enumerate(self.names) if n not in POSTURE)

    def to_list(self) -> list[str]:
        return list(self.names)


@dataclass
class BehaviourSpan:
    """One behaviour, localised in time, as produced by inference."""

    behaviour: str
    start: float
    end: float
    score: float
    peak: float = field(default=0.0)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "behaviour": self.behaviour,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "score": round(self.score, 4),
            "peak": round(self.peak, 4),
        }

    def __str__(self) -> str:
        return (
            f"{self.start:7.2f}s -> {self.end:7.2f}s  "
            f"{self.behaviour:<16} {self.score:.2f}"
        )
