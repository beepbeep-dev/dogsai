"""Turning detected behaviour into a read on how the dog seems to feel.

Read this before trusting the output
------------------------------------
**No model can tell you what a dog feels.** Internal emotional state is not
observable, not from video and not by any other remote means. What *is*
observable is body language, and what animal-welfare science does with it is map
observable signals onto two axes rather than onto named emotions:

* **arousal** — how activated the animal is, from asleep to frantic;
* **valence** — whether the situation appears positive or negative to it.

That two-axis "core affect" framing is standard in the field (Mendl, Burman &
Paul, 2010) precisely because it is what the evidence supports: you can see that
a dog is highly aroused and that the signals lean negative, which is useful, and
you cannot see whether it feels "anxious" versus "frustrated", which is a
distinction the video does not contain.

So this module produces a *body-language read*: two numbers, a plain-English
description of the quadrant they fall in, the evidence behind it, and an explicit
statement of what the model could not see. It is a screening aid — "this clip is
worth a human's attention" — not a diagnosis.

Three specific limits worth stating up front, because they are the ways this kind
of output misleads people:

1. **Context is invisible.** A dog panting hard is thermoregulating in the sun
   and stressed at the vet, and the pixels look the same. The single largest
   determinant of what a behaviour means is the situation it happens in.
2. **Displacement signals are ambiguous by nature.** A yawn is a stress signal
   *out of context* and is otherwise just a tired dog. That is why yawning here
   contributes mild evidence, not a verdict.
3. **The read is capped by the taxonomy.** A model trained on five behaviours
   cannot see tail position, ear set, lip licking or whale eye — the signals a
   behaviourist would weight most heavily. :attr:`AffectReading.coverage` reports
   how much affect-relevant evidence actually existed, and the summary says so
   out loud when it is thin.

If a dog's behaviour concerns you, the answer is a vet or a qualified behaviourist
who can see the context. Nothing here substitutes for that.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .labels import BehaviourSpan


@dataclass(frozen=True)
class Contribution:
    """How one behaviour informs the two affect axes.

    ``valence`` runs -1 (appears negative) to +1 (appears positive); ``arousal``
    runs 0 (inert) to 1 (highly activated). ``weight`` is how much this signal is
    trusted relative to the others — deliberately low for anything ambiguous.
    """

    valence: float
    arousal: float
    weight: float = 1.0
    note: str = ""
    stress_signal: bool = False


# Weights are intentionally conservative. The unambiguous postures (lying,
# running) carry arousal information but say little about valence, so their
# valence weight is small. The classic displacement behaviours carry negative
# valence evidence but are individually weak, so no single one can swing the read.
AFFECT_MAP: dict[str, Contribution] = {
    # -- posture / locomotion: mostly arousal information -----------------
    "lying_down": Contribution(
        valence=0.15, arousal=0.05, weight=0.8,
        note="settled, low activation — most often a resting dog",
    ),
    "sitting": Contribution(
        valence=0.05, arousal=0.25, weight=0.4,
        note="calm but attentive",
    ),
    "standing": Contribution(
        valence=0.0, arousal=0.35, weight=0.2,
        note="neutral baseline posture",
    ),
    "walking": Contribution(
        valence=0.1, arousal=0.45, weight=0.4,
        note="relaxed movement",
    ),
    "trotting": Contribution(
        valence=0.1, arousal=0.65, weight=0.4,
        note="purposeful movement, moderate activation",
    ),
    "running": Contribution(
        valence=0.15, arousal=0.9, weight=0.5,
        note="high activation; valence depends entirely on context "
             "(zoomies and fleeing look similar from outside)",
    ),
    # -- clearly positive-leaning -----------------------------------------
    "playing": Contribution(
        valence=0.85, arousal=0.8, weight=1.4,
        note="play is one of the more reliable positive-affect indicators in dogs",
    ),
    "tail_wagging": Contribution(
        valence=0.5, arousal=0.6, weight=0.7,
        note="loose wagging leans positive, but a stiff high wag can signal "
             "arousal or threat — this model cannot tell the two apart",
    ),
    "chewing": Contribution(
        valence=0.5, arousal=0.4, weight=0.8,
        note="chewing and tugging are usually contented, self-directed activity",
    ),
    "eating_drinking": Contribution(
        valence=0.45, arousal=0.35, weight=0.9,
        note="willingness to eat is a broadly positive welfare indicator",
    ),
    "stretching": Contribution(
        valence=0.35, arousal=0.2, weight=0.6,
        note="typically a comfortable, unguarded dog",
    ),
    "rolling": Contribution(
        valence=0.4, arousal=0.5, weight=0.5,
        note="rolling usually reads as relaxed or playful",
    ),
    "sniffing": Contribution(
        valence=0.3, arousal=0.4, weight=0.7,
        note="exploratory sniffing is enriching behaviour and leans positive; "
             "frantic ground-sniffing can also be displacement",
    ),
    "digging": Contribution(
        valence=0.2, arousal=0.7, weight=0.4,
        note="often normal play or instinct; can be boredom or escape-motivated",
    ),
    "jumping": Contribution(
        valence=0.4, arousal=0.85, weight=0.6,
        note="excited greeting or play, high activation",
    ),
    "eliminating": Contribution(
        valence=0.0, arousal=0.3, weight=0.2,
        note="normal maintenance behaviour; carries little affect information",
    ),
    # -- stress / appeasement signals (weak individually, telling together)
    "yawning": Contribution(
        valence=-0.35, arousal=0.4, weight=0.6, stress_signal=True,
        note="out-of-context yawning is a recognised appeasement/displacement "
             "signal — but a tired dog also just yawns",
    ),
    "scratching": Contribution(
        valence=-0.3, arousal=0.45, weight=0.5, stress_signal=True,
        note="can be a displacement behaviour under mild stress; can equally be "
             "fleas, allergies or dry skin — a vet question, not a mood question",
    ),
    "shaking_off": Contribution(
        valence=-0.2, arousal=0.5, weight=0.4, stress_signal=True,
        note="a shake-off away from water often marks a stress 'reset' after "
             "a tense moment",
    ),
    "barking": Contribution(
        valence=-0.3, arousal=0.85, weight=0.8, stress_signal=True,
        note="high arousal; alarm, frustration, demand and excitement all bark, "
             "and audio would be needed to tell them apart",
    ),
    "alert_freeze": Contribution(
        valence=-0.55, arousal=0.7, weight=1.1, stress_signal=True,
        note="stillness with a hard stare is a meaningful warning sign — dogs "
             "commonly freeze before escalating",
    ),
}

# Signals a behaviourist would weight heavily that a clip-level behaviour
# classifier does not represent at all. Named explicitly so the output can admit
# what it is blind to instead of implying completeness.
UNSEEN_SIGNALS: tuple[str, ...] = (
    "tail position (high/tucked) as distinct from wagging",
    "ear set and facial tension",
    "lip licking, nose licking",
    "whale eye (visible sclera)",
    "weight distribution / leaning away",
    "piloerection (raised hackles)",
    "panting rate, and whether it is heat or stress",
    "vocalisation tone (growl vs whine vs bark) — needs audio",
)


@dataclass
class AffectReading:
    """A body-language read, with its own uncertainty attached."""

    valence: float
    arousal: float
    label: str
    confidence: float
    coverage: float
    evidence: list[tuple[str, float, str]] = field(default_factory=list)
    stress_signals: list[str] = field(default_factory=list)
    unseen: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def needs_human_review(self) -> bool:
        """Whether a person should watch this clip themselves.

        Errs toward yes: a negative read, or two-plus independent stress signals,
        or high arousal with negative lean.
        """
        return (
            self.valence < -0.15
            or len(self.stress_signals) >= 2
            or (self.arousal > 0.7 and self.valence < 0.0)
        )

    def to_dict(self) -> dict:
        return {
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "coverage": round(self.coverage, 3),
            "needs_human_review": self.needs_human_review,
            "evidence": [
                {"behaviour": b, "seconds": round(s, 2), "note": n}
                for b, s, n in self.evidence
            ],
            "stress_signals": self.stress_signals,
            "not_observable_by_this_model": self.unseen,
            "notes": self.notes,
            "disclaimer": (
                "This is an inference from observable body language, not a "
                "measurement of emotion. Behaviour meaning is context-dependent. "
                "For any welfare or behaviour concern, consult a veterinarian or "
                "qualified behaviourist."
            ),
        }

    def render(self) -> str:
        """Human-readable report, uncertainty first."""
        bar_v = _bar(self.valence, signed=True)
        bar_a = _bar(self.arousal, signed=False)
        lines = [
            f"how the dog seems:  {self.label}",
            "",
            f"  valence  {bar_v}  {self.valence:+.2f}   (negative <-> positive)",
            f"  arousal  {bar_a}  {self.arousal:.2f}   (calm <-> activated)",
            f"  confidence {self.confidence:.0%}   evidence coverage {self.coverage:.0%}",
        ]
        if self.evidence:
            lines += ["", "  based on:"]
            for behaviour, seconds, note in self.evidence:
                lines.append(f"    {behaviour:<16} {seconds:5.1f}s  {note}")
        if self.stress_signals:
            lines += [
                "",
                "  possible stress / appeasement signals: "
                + ", ".join(self.stress_signals),
            ]
        if self.notes:
            lines += [""] + [f"  note: {n}" for n in self.notes]
        if self.needs_human_review:
            lines += ["", "  -> worth watching yourself; the signals lean negative."]
        lines += [
            "",
            "  what this model cannot see: " + "; ".join(self.unseen[:4]) + ".",
            "  This is a read on body language, not a measurement of emotion, and",
            "  behaviour means different things in different contexts. If you are",
            "  worried about your dog, ask a vet or a qualified behaviourist.",
        ]
        return "\n".join(lines)


def _bar(value: float, signed: bool, width: int = 21) -> str:
    """Render a value as a text gauge."""
    if signed:
        centre = width // 2
        position = int(round(centre + value * centre))
        position = max(0, min(width - 1, position))
        cells = ["-"] * width
        cells[centre] = "|"
        cells[position] = "#"
        return "[" + "".join(cells) + "]"
    filled = max(0, min(width, int(round(value * width))))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _quadrant(valence: float, arousal: float, coverage: float) -> str:
    """Name the region of the valence/arousal plane, hedged when evidence is thin."""
    if coverage < 0.15:
        return "not enough to say"
    strong_v = abs(valence) > 0.3
    if arousal >= 0.6:
        if valence >= 0.25:
            return "excited and enjoying it" if strong_v else "excited"
        if valence <= -0.2:
            return "agitated or stressed"
        return "highly activated, valence unclear"
    if arousal <= 0.3:
        if valence >= 0.2:
            return "relaxed and content"
        if valence <= -0.2:
            return "withdrawn or uncomfortable"
        return "quiet and settled"
    if valence >= 0.25:
        return "comfortable and engaged"
    if valence <= -0.2:
        return "uneasy"
    return "neutral"


def read_affect(
    spans: list[BehaviourSpan],
    duration: float,
    known_behaviours: list[str] | None = None,
    min_score: float = 0.0,
) -> AffectReading:
    """Aggregate detected behaviour spans into a body-language read.

    Contributions are weighted by ``duration x detection score x signal weight``,
    so a confident 6-second play bout outweighs a marginal half-second blip. The
    resulting confidence deliberately shrinks when little affect-relevant
    behaviour was detected, when detections were weak, or when the model's
    taxonomy contains few affect-bearing behaviours in the first place.

    ``min_score`` defaults to 0 because spans arrive *already thresholded* by
    :class:`~dogsai.predict.BehaviourPredictor`, against per-class thresholds
    fitted on validation data. Applying a second absolute floor here would
    double-filter, and would break multiclass outright: a softmax winner over
    five classes sits around 0.3, so any fixed floor tuned for sigmoid scores
    silently discards every span and reports "not enough to say" on a video the
    model actually understood. Score is used to *weight* evidence, not to gate it.
    """
    duration = max(duration, 1e-6)
    per_behaviour: dict[str, float] = {}
    total_weight = 0.0
    valence_sum = 0.0
    arousal_sum = 0.0
    stress: list[str] = []
    covered = 0.0

    for span in spans:
        contribution = AFFECT_MAP.get(span.behaviour)
        if contribution is None or span.score < min_score or span.score <= 0:
            continue
        weight = span.duration * span.score * contribution.weight
        if weight <= 0:
            continue
        per_behaviour[span.behaviour] = per_behaviour.get(span.behaviour, 0.0) + span.duration
        valence_sum += contribution.valence * weight
        arousal_sum += contribution.arousal * weight
        total_weight += weight
        covered += span.duration
        if contribution.stress_signal and span.behaviour not in stress:
            stress.append(span.behaviour)

    if total_weight <= 0:
        return AffectReading(
            valence=0.0,
            arousal=0.0,
            label="not enough to say",
            confidence=0.0,
            coverage=0.0,
            unseen=list(UNSEEN_SIGNALS),
            notes=[
                "no behaviour was detected confidently enough to say anything "
                "about how the dog seems"
            ],
        )

    valence = max(-1.0, min(1.0, valence_sum / total_weight))
    arousal = max(0.0, min(1.0, arousal_sum / total_weight))
    coverage = min(1.0, covered / duration)

    evidence = sorted(
        (
            (name, seconds, AFFECT_MAP[name].note)
            for name, seconds in per_behaviour.items()
        ),
        key=lambda item: -item[1],
    )

    notes: list[str] = []
    # Confidence starts from how much of the clip carried affect-relevant
    # evidence, then is knocked down for every reason to distrust it.
    confidence = min(1.0, coverage * 1.2)
    if known_behaviours:
        informative = [b for b in known_behaviours if b in AFFECT_MAP]
        ratio = len(informative) / max(1, len(known_behaviours))
        # A 5-class model simply has less to go on than an 18-class one.
        breadth = min(1.0, len(informative) / 8.0)
        confidence *= 0.45 + 0.55 * breadth
        if len(informative) < 6:
            notes.append(
                f"this model only recognises {len(informative)} affect-relevant "
                f"behaviour(s) ({', '.join(informative)}), so the read is coarse"
            )
        if ratio < 1.0:
            missing = [b for b in known_behaviours if b not in AFFECT_MAP]
            notes.append(
                f"detected behaviours with no affect mapping were ignored: "
                f"{', '.join(missing)}"
            )
    if len(evidence) == 1:
        confidence *= 0.7
        notes.append("only one behaviour type was detected — a single signal is weak evidence")
    if stress:
        notes.append(
            "stress-signal behaviours are ambiguous in isolation and depend "
            "heavily on context"
        )
    confidence = max(0.0, min(1.0, confidence))

    return AffectReading(
        valence=valence,
        arousal=arousal,
        label=_quadrant(valence, arousal, coverage),
        confidence=confidence,
        coverage=coverage,
        evidence=evidence,
        stress_signals=stress,
        unseen=list(UNSEEN_SIGNALS),
        notes=notes,
    )
