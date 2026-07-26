"""What to do about it.

Why this module is rules and not a model
---------------------------------------
Everything else in this package that makes a judgement is either learned from data
or computed from signal. This is neither, deliberately: **there is no dataset of
"dog did X, owner should do Y".** Nothing to train on means nothing to learn, and a
model that produced advice anyway would be generating plausible-sounding text with
no grounding — the worst possible property for guidance someone might act on.

So the suggestions here are a curated mapping from observed behaviour and affect
state to conventional, low-risk husbandry advice. They are the kind of thing a
sensible dog book says, encoded so the output can be traced to the observation that
triggered it.

Scope limits, which are strict on purpose:

* **Nothing medical.** No dosages, no diagnoses, no treatments. Where a signal is
  potentially clinical — a yelp, persistent scratching, repeated unproductive
  straining — the advice is "have a vet look at this", full stop.
* **Nothing aversive.** No corrections, no punishment, no dominance framing. Aside
  from the welfare case against them, they make fear- and pain-driven behaviour
  worse, and this module cannot tell what is driving anything.
* **Context-blind by construction.** The suggestions are conditional and hedged
  because the model cannot see the situation, and the situation is what determines
  what a behaviour means.

If something concerns you, the advice is always the same and it is not in this
file: ask a vet or a qualified behaviourist who can see the whole picture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .affect import AffectReading
from .audio import NOT_THE_DOG, AudioReading
from .labels import BehaviourSpan


@dataclass(frozen=True)
class Suggestion:
    """One piece of advice, and what prompted it."""

    text: str
    because: str
    priority: int = 2          # 1 = act now, 2 = worth doing, 3 = optional
    veterinary: bool = False

    def to_dict(self) -> dict:
        return {
            "advice": self.text,
            "because": self.because,
            "priority": self.priority,
            "veterinary": self.veterinary,
        }

    def __str__(self) -> str:
        mark = {1: "!!", 2: " -", 3: "  "}.get(self.priority, " -")
        return f"{mark} {self.text}\n     (because: {self.because})"


# Behaviour-triggered advice. Keyed on the taxonomy so it stays in step with it.
_BEHAVIOUR_ADVICE: dict[str, tuple[Suggestion, ...]] = {
    "playing": (
        Suggestion(
            "Good — let it continue, and watch for the point where excitement tips "
            "into being unable to settle.",
            "play detected", 3,
        ),
    ),
    "chewing": (
        Suggestion(
            "Check what is being chewed is actually chew-safe: no small pieces, "
            "nothing that can splinter or be swallowed whole.",
            "sustained chewing", 2,
        ),
    ),
    "eating_drinking": (
        Suggestion(
            "Nothing needed. A dog eating willingly is a good sign.",
            "eating or drinking", 3,
        ),
    ),
    "eliminating": (
        Suggestion(
            "If straining goes on without result, or you see blood, treat that as a "
            "same-day vet question.",
            "eliminating", 2, veterinary=True,
        ),
    ),
    "yawning": (
        Suggestion(
            "Look at what changed just before the yawn. Out of context, repeated "
            "yawning is often mild stress rather than tiredness — if something in "
            "the room prompted it, give the dog more distance from it.",
            "yawning, a possible appeasement signal", 2,
        ),
    ),
    "scratching": (
        Suggestion(
            "Persistent scratching is far more often skin, fleas or allergies than "
            "mood. Worth a vet check rather than a behavioural fix.",
            "repeated scratching", 2, veterinary=True,
        ),
    ),
    "digging": (
        Suggestion(
            "Usually normal, and often boredom. More sniffing walks and something to "
            "work on beats trying to stop the digging itself.",
            "digging", 3,
        ),
    ),
    "sniffing": (
        Suggestion(
            "Let it finish. Sniffing is genuine enrichment and rushing a dog past it "
            "removes most of the value of a walk.",
            "exploratory sniffing", 3,
        ),
    ),
    "alert_freeze": (
        Suggestion(
            "Freezing with a hard stare often comes before escalation. Calmly "
            "increase the distance between the dog and whatever it is fixed on; do "
            "not reach for the dog or lean over it.",
            "alert freeze — a pre-escalation signal", 1,
        ),
    ),
    "barking": (
        Suggestion(
            "Work out what it is directed at before responding to the noise itself.",
            "barking", 2,
        ),
    ),
    "shaking_off": (
        Suggestion(
            "A shake-off away from water often marks the end of a tense moment — a "
            "useful cue that whatever just happened was a bit much.",
            "shake-off, often a stress reset", 3,
        ),
    ),
    "stretching": (
        Suggestion("Nothing needed — a relaxed, comfortable dog.", "stretching", 3),
    ),
    "lying_down": (
        Suggestion("Nothing needed. Let a resting dog rest.", "resting", 3),
    ),
}

# Vocalisation-triggered advice, which tends to be more urgent.
_VOICE_ADVICE: dict[str, tuple[Suggestion, ...]] = {
    "yelp": (
        Suggestion(
            "A yelp usually means sudden pain or a real fright. Check the dog over "
            "gently — feet, joints, anywhere it flinches from — and get it seen if it "
            "recurs, limps, or does not settle.",
            "a yelp was heard", 1, veterinary=True,
        ),
    ),
    "growl": (
        Suggestion(
            "Do not punish the growl. It is a warning, and suppressing it removes the "
            "warning without removing the reason — a dog that has learned not to "
            "growl is a dog that bites without notice. Give it space and work out "
            "what it was asking for distance from.",
            "a growl was heard", 1,
        ),
    ),
    "bark_alarm": (
        Suggestion(
            "Identify what triggered it and reduce the exposure — distance, a barrier, "
            "or blocking the line of sight — rather than trying to out-shout it.",
            "low, harsh, repeated barking", 2,
        ),
    ),
    "bark_excited": (
        Suggestion(
            "Excitement barking. Wait for a pause before giving attention, so the "
            "quiet is what gets rewarded.",
            "high, clear barking", 3,
        ),
    ),
    "whine": (
        Suggestion(
            "Whining is usually a request. Check the obvious needs first — water, "
            "the toilet, something out of reach, somewhere too hot or too cold.",
            "whining", 2,
        ),
    ),
    "howl": (
        Suggestion(
            "Howling often means being alone. If it happens whenever you leave, that "
            "pattern is worth raising with a behaviourist rather than waiting out.",
            "howling", 2,
        ),
    ),
    "pant": (
        Suggestion(
            "Check for heat first — shade, water, and a cooler spot. If it is not "
            "warm and the dog is panting anyway, consider stress or pain.",
            "panting", 2, veterinary=True,
        ),
    ),
}


@dataclass
class Advice:
    """A prioritised, traceable set of suggestions."""

    suggestions: list[Suggestion] = field(default_factory=list)
    summary: str = ""
    urgent: bool = False

    @property
    def veterinary(self) -> list[Suggestion]:
        return [s for s in self.suggestions if s.veterinary]

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "urgent": self.urgent,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "disclaimer": (
                "General guidance derived from the detected signals by a fixed rule "
                "set, not a trained model and not veterinary advice. Behaviour is "
                "context-dependent and this has no access to the context. For "
                "anything that worries you, consult a veterinarian or a qualified "
                "behaviourist."
            ),
        }

    def render(self) -> str:
        if not self.suggestions:
            return "what to do: nothing to suggest — too little was detected."
        lines = ["what to do", "=" * 10, ""]
        if self.summary:
            lines += [f"  {self.summary}", ""]
        for suggestion in self.suggestions:
            lines.append(f"  {suggestion}")
        lines += [
            "",
            "  These are general suggestions from a fixed rule set, not a trained",
            "  model and not veterinary advice. Behaviour means different things in",
            "  different situations and this cannot see yours. Anything that worries",
            "  you is a question for a vet or a qualified behaviourist.",
        ]
        return "\n".join(lines)


def advise(
    spans: list[BehaviourSpan] | None = None,
    audio: AudioReading | None = None,
    affect: AffectReading | None = None,
    min_duration: float = 0.5,
    max_suggestions: int = 6,
) -> Advice:
    """Assemble advice from the detected behaviour, voice and affect state.

    Ordered by priority so the thing that might matter is first, deduplicated, and
    capped — a wall of twenty suggestions is one nobody reads.
    """
    spans = spans or []
    collected: list[Suggestion] = []
    seen: set[str] = set()

    def add(suggestion: Suggestion) -> None:
        if suggestion.text not in seen:
            seen.add(suggestion.text)
            collected.append(suggestion)

    # Vocalisations first: they carry the more urgent signals.
    if audio is not None:
        for event in audio.events:
            if event.kind in NOT_THE_DOG:
                continue
            for suggestion in _VOICE_ADVICE.get(event.kind, ()):
                add(suggestion)

    totals: dict[str, float] = {}
    for span in spans:
        totals[span.behaviour] = totals.get(span.behaviour, 0.0) + span.duration
    for behaviour, seconds in sorted(totals.items(), key=lambda kv: -kv[1]):
        if seconds < min_duration:
            continue
        for suggestion in _BEHAVIOUR_ADVICE.get(behaviour, ()):
            add(suggestion)

    # State-level advice, from the two axes rather than any single behaviour.
    if affect is not None and affect.confidence > 0.2:
        if affect.valence < -0.2 and affect.arousal > 0.6:
            add(Suggestion(
                "The signals point to being worked up and unhappy at the same time. "
                "Reduce what is driving it if you can identify it, and give the dog "
                "somewhere quiet it can choose to go.",
                "negative valence with high arousal", 1,
            ))
        elif affect.valence < -0.2 and affect.arousal < 0.35:
            add(Suggestion(
                "Quiet and withdrawn. Worth watching for whether it is just rest or "
                "the dog opting out — a change from its normal baseline matters more "
                "than the absolute state.",
                "negative valence with low arousal", 2,
            ))
        if len(affect.stress_signals) >= 2:
            add(Suggestion(
                "Two or more possible stress signals in one short clip. Any one of "
                "them means little alone; together they are worth taking seriously. "
                "Look at what is common to the moments they happened in.",
                f"multiple stress signals: {', '.join(affect.stress_signals)}", 2,
            ))

    collected.sort(key=lambda s: s.priority)
    collected = collected[:max_suggestions]
    urgent = any(s.priority == 1 for s in collected)

    if not collected:
        summary = ""
    elif urgent:
        summary = "There is something here worth your attention now."
    elif any(s.veterinary for s in collected):
        summary = "Mostly ordinary, with one thing worth mentioning to a vet."
    else:
        summary = "Nothing alarming — routine suggestions only."

    return Advice(suggestions=collected, summary=summary, urgent=urgent)
