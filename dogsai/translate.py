"""Rendering what the dog is communicating into plain English.

What this is, in one paragraph
------------------------------
Dogs communicate constantly — posture, movement, and voice are all signals aimed
at whoever is watching. What they do *not* have is words. So this module does not
decode speech; there is no wording inside a bark to recover. What it does is take
the signals the rest of the package detected — behaviour spans from the video
model, vocalisation types from the audio analysis, and the arousal/valence read
built on both — and phrase them the way the dog would put it if it could talk.

That is a real translation in the useful sense: the *meaning* is carried across
from one system of signals into another. It is not a real translation in the sense
of decoding a language, because there is no language there. Both halves of that
matter. The output is grounded — every line traces back to a specific detection
you can inspect — but the first person is a presentation choice, not a claim that
the dog composed the sentence.

Confidence is inherited honestly. If the video model was unsure and the dog was
silent, the translation says so instead of inventing dialogue. It never fabricates
an utterance with no signal behind it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .affect import AffectReading
from .audio import NOT_THE_DOG, AudioReading, VocalEvent
from .labels import BehaviourSpan

# What each behaviour would sound like in the first person. Several options per
# behaviour, chosen deterministically from the detection so repeated runs on the
# same clip produce the same text.
_BEHAVIOUR_LINES: dict[str, tuple[str, ...]] = {
    "playing": (
        "This is the best thing that has ever happened. Again!",
        "I'm having a brilliant time. Do not stop doing this.",
        "Come on, come on, keep going!",
    ),
    "chewing": (
        "I've got my thing and I'm working on it. It's mine.",
        "This is very satisfying and I'd like to be left to it.",
    ),
    "eating_drinking": (
        "Food. This is the important part of the day.",
        "I'm eating. Please don't move the bowl.",
    ),
    "eliminating": (
        "Give me a minute, I'm busy.",
        "Bit of privacy would be nice.",
    ),
    "yawning": (
        "I'm either sleepy or a bit unsettled — even I'm not sure which.",
        "Long day. Or I'm not quite comfortable here.",
    ),
    "lying_down": (
        "I'm comfortable and nothing needs doing.",
        "This is my spot now. I've settled.",
    ),
    "sitting": (
        "I'm waiting. Something is supposed to happen, isn't it?",
        "Sitting nicely. Watching you. Any minute now.",
    ),
    "standing": (
        "I'm just here, keeping an eye on things.",
        "Standing by. Nothing much going on.",
    ),
    "walking": ("Off we go. Nothing urgent.", "Having a wander."),
    "trotting": ("I've somewhere to be.", "Places to go, things to check."),
    "running": ("FULL SPEED. No notes.", "Fast! Fast! This is what legs are for!"),
    "jumping": (
        "Look at me look at me look at me!",
        "Up! I'm up here! Hello!",
    ),
    "sniffing": (
        "Hold on — something's been here.",
        "There's a whole story on this ground and I'm reading it.",
    ),
    "digging": (
        "There is definitely something under here.",
        "Nearly through. Don't interrupt.",
    ),
    "tail_wagging": ("I like this. I like you.", "Yes. Yes to all of this."),
    "scratching": ("Itchy. Dealing with it.", "That spot again."),
    "shaking_off": ("Right — shaking that off.", "Resetting. Moving on."),
    "rolling": ("This is a very good spot.", "Getting the smell on me. Excellent."),
    "stretching": ("Loooong. Good.", "Just working the stiffness out."),
    "alert_freeze": (
        "I've seen something. I'm not moving. I'm watching it.",
        "Something's there. Very still now.",
    ),
    "barking": ("Oi. Attention, please.", "Hey! Hey!"),
}

# Vocalisations, which usually carry the more urgent message.
_VOICE_LINES: dict[str, tuple[str, ...]] = {
    "bark_excited": (
        "Hey! Hey! Come and look at this!",
        "You're here! You're here! This is excellent!",
    ),
    "bark_alarm": (
        "Something's out there and I don't like it. Back off.",
        "I've noticed that and I want it to stop. Warning you.",
    ),
    "bark": ("Oi! Attention over here.", "Hey. Listen."),
    "growl": (
        "That's close enough. I mean it.",
        "I'm asking you nicely to stop. This is the nice version.",
    ),
    "whine": (
        "Please. I need something and I can't sort it myself.",
        "I'm not happy about this and I'd like your help.",
    ),
    "howl": (
        "Anyone out there? I'm over here!",
        "Hello? Hello? I don't like being the only one.",
    ),
    "yelp": ("Ow! That hurt!", "Ah! Something's wrong!"),
    "pant": ("Warm. Getting my breath back.", "Bit hot. Bit much."),
}

# Headline summaries by (arousal band, valence band).
_HEADLINES: dict[tuple[str, str], tuple[str, ...]] = {
    ("high", "positive"): (
        "I am having a wonderful time and I need you to know about it.",
        "Everything is brilliant and I have a lot of energy about it.",
    ),
    ("high", "neutral"): (
        "A lot is happening and I'm right in the middle of it.",
        "I'm very awake and very busy.",
    ),
    ("high", "negative"): (
        "I'm worked up and I'm not happy. Something needs to change.",
        "I don't like this and I'm telling you loudly.",
    ),
    ("medium", "positive"): (
        "I'm content and getting on with something I enjoy.",
        "Things are good. I'm busy in a nice way.",
    ),
    ("medium", "neutral"): (
        "I'm just going about my business.",
        "Nothing much either way. Pottering.",
    ),
    ("medium", "negative"): (
        "Something's not quite right and it's on my mind.",
        "I'm a bit uneasy about this.",
    ),
    ("low", "positive"): (
        "I'm relaxed and everything is fine.",
        "Comfortable, settled, no complaints.",
    ),
    ("low", "neutral"): ("Quiet. Nothing to report.", "Just resting."),
    ("low", "negative"): (
        "I've gone quiet and I'm not enjoying myself.",
        "I'd rather be somewhere else.",
    ),
}


def _pick(options: tuple[str, ...], seed: float) -> str:
    """Choose deterministically from ``options`` using a numeric feature as key."""
    if not options:
        return ""
    return options[int(abs(seed) * 1000) % len(options)]


@dataclass
class Utterance:
    """One line of the translation, and the detection it came from."""

    start: float
    end: float
    text: str
    confidence: float
    source: str            # "behaviour" | "voice"
    basis: str             # the detection this came from
    urgent: bool = False

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "source": self.source,
            "basis": self.basis,
            "urgent": self.urgent,
        }

    def __str__(self) -> str:
        mark = "!" if self.urgent else " "
        return f"{mark} {self.start:6.2f}s  \"{self.text}\"   [{self.basis}]"


@dataclass
class Translation:
    """The dog's side of the conversation, as far as it can be reconstructed."""

    headline: str
    utterances: list[Utterance] = field(default_factory=list)
    confidence: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0
    urgent: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    heard_voice: bool = False
    saw_behaviour: bool = False

    def to_dict(self) -> dict:
        return {
            "headline": self.headline,
            "confidence": round(self.confidence, 3),
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "utterances": [u.to_dict() for u in self.utterances],
            "urgent": self.urgent,
            "notes": self.notes,
            "used_video": self.saw_behaviour,
            "used_audio": self.heard_voice,
            "disclaimer": (
                "Dogs communicate but do not use words, so this is a rendering of "
                "detected signals — behaviour and vocalisation type — into English, "
                "not decoded speech. Every line traces back to a specific detection; "
                "the first person is presentation, not a claim about what the dog "
                "composed. Behaviour meaning is context-dependent, and this has no "
                "access to the context."
            ),
        }

    def render(self, width: int = 72) -> str:
        lines = ["what your dog is telling you", "=" * min(width, 40), ""]
        lines.append(f'  "{self.headline}"')
        lines.append("")
        if self.utterances:
            lines.append("  moment by moment:")
            for utterance in self.utterances:
                lines.append(f"  {utterance}")
            lines.append("")
        channels = []
        if self.saw_behaviour:
            channels.append("what it did")
        if self.heard_voice:
            channels.append("what it said out loud")
        lines.append(
            f"  read from: {' and '.join(channels) if channels else 'not much to go on'}"
        )
        lines.append(
            f"  confidence {self.confidence:.0%}   "
            f"valence {self.valence:+.2f}   arousal {self.arousal:.2f}"
        )
        if self.urgent:
            lines += ["", f"  worth a closer look: {', '.join(self.urgent)}"]
        if self.notes:
            lines += [""] + [f"  note: {n}" for n in self.notes]
        lines += [
            "",
            "  How to read this: dogs communicate, but not in words — so these are",
            "  its signals put into English, not speech decoded. Each line traces to",
            "  a detection you can check. The same signal means different things in",
            "  different situations, and this has no idea what the situation was.",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


def _band(value: float, low: float, high: float) -> str:
    if value >= high:
        return "high"
    if value <= low:
        return "low"
    return "medium"


def _valence_band(value: float) -> str:
    if value >= 0.2:
        return "positive"
    if value <= -0.15:
        return "negative"
    return "neutral"


def translate(
    spans: list[BehaviourSpan] | None = None,
    audio: AudioReading | None = None,
    affect: AffectReading | None = None,
    duration: float = 0.0,
    min_span: float = 0.4,
) -> Translation:
    """Turn detections into a first-person account of what the dog is saying.

    Vocalisations are given priority over posture when both are present in the
    same moment: a growl during play is the more informative signal, and a
    translation that reported only "I'm having fun" would be actively misleading.
    """
    spans = spans or []
    utterances: list[Utterance] = []
    urgent: list[str] = []
    notes: list[str] = []

    # -- what it did ----------------------------------------------------
    for span in spans:
        if span.duration < min_span:
            continue
        options = _BEHAVIOUR_LINES.get(span.behaviour)
        if not options:
            continue
        utterances.append(
            Utterance(
                start=span.start,
                end=span.end,
                text=_pick(options, span.start + span.score),
                confidence=span.score,
                source="behaviour",
                basis=f"{span.behaviour} {span.start:.1f}-{span.end:.1f}s",
            )
        )

    # -- what it said ---------------------------------------------------
    voice_events: list[VocalEvent] = []
    if audio is not None:
        voice_events = [e for e in audio.events if e.kind not in NOT_THE_DOG]
        for event in voice_events:
            options = _VOICE_LINES.get(event.kind)
            if not options:
                continue
            utterances.append(
                Utterance(
                    start=event.start,
                    end=event.end,
                    text=_pick(options, event.start + event.f0 / 1000.0),
                    confidence=event.confidence,
                    source="voice",
                    basis=f"{event.kind} {event.start:.1f}s",
                    urgent=event.concerning,
                )
            )
            if event.concerning and event.kind not in urgent:
                urgent.append(event.kind)
        spoken = sum(1 for e in audio.events if e.kind == "speech")
        if spoken:
            notes.append(
                f"{spoken} sound(s) were a person talking, not the dog, and were ignored"
            )

    # Voice first when two lines land at the same moment.
    utterances.sort(key=lambda u: (u.start, 0 if u.source == "voice" else 1))

    # Drop a behaviour line that repeats the previous one back to back — the
    # sliding window fragments long behaviours, and the translation should read
    # like speech rather than a stutter.
    deduped: list[Utterance] = []
    for utterance in utterances:
        if (
            deduped
            and utterance.text == deduped[-1].text
            and utterance.start - deduped[-1].end < 1.5
        ):
            deduped[-1].end = max(deduped[-1].end, utterance.end)
            continue
        deduped.append(utterance)
    utterances = deduped

    # -- combine the two channels into one state ------------------------
    parts: list[tuple[float, float, float]] = []  # (valence, arousal, weight)
    if affect is not None and affect.confidence > 0:
        parts.append((affect.valence, affect.arousal, max(0.2, affect.confidence)))
    if audio is not None and audio.confidence > 0:
        # Voice is weighted a little above the visual read when present: it is the
        # channel the dog is actively using to tell you something.
        parts.append((audio.valence, audio.arousal, max(0.2, audio.confidence) * 1.3))

    if parts:
        total = sum(w for _, _, w in parts)
        valence = sum(v * w for v, _, w in parts) / total
        arousal = sum(a * w for _, a, w in parts) / total
        confidence = min(1.0, total / 2.2)
    else:
        valence = arousal = confidence = 0.0

    # An urgent vocalisation must not be averaged away by a cheerful video read.
    # A yelp or a growl is high-information and asymmetric: the cost of missing
    # one is much higher than the cost of flagging a clip that turned out fine.
    # Where the two channels disagree, say so rather than splitting the difference
    # into a number that represents neither.
    urgent_events = [e for e in voice_events if e.concerning]
    if urgent_events:
        arousal = max(arousal, max(e.arousal for e in urgent_events))
        # Test the *channels* against each other, not the blend. Two signals
        # pointing opposite ways can average to a neutral-looking number, and
        # reporting that number alone hides the disagreement completely — which is
        # the case most worth telling someone about.
        looks_positive = affect is not None and affect.confidence > 0 and affect.valence > 0.2
        if looks_positive or valence > 0.15:
            kinds = ", ".join(sorted({e.kind for e in urgent_events}))
            notes.append(
                "the video and the audio disagree: what the dog was doing looks "
                f"positive, but it made a sound ({kinds}) that does not. The sound "
                f"is the more urgent signal, so treat the summary above as the "
                f"cautious reading"
            )
            valence = min(valence, 0.15)

    saw = bool(spans)
    heard = bool(voice_events)
    if not saw and not heard:
        return Translation(
            headline="I'm not giving you much to go on right now.",
            confidence=0.0,
            notes=notes + [
                "no behaviour was detected confidently and no dog vocalisation was "
                "heard, so there is nothing to translate"
            ],
        )
    if not heard:
        notes.append("the dog was silent, so this is read from movement alone")
    if not saw:
        notes.append("no behaviour was confidently detected, so this is from the voice alone")

    headline_options = _HEADLINES[(_band(arousal, 0.3, 0.6), _valence_band(valence))]
    headline = _pick(headline_options, valence + arousal)
    # An urgent vocalisation overrides a cheerful summary.
    if any(u.urgent for u in utterances):
        loudest = max((u for u in utterances if u.urgent), key=lambda u: u.confidence)
        headline = loudest.text

    if affect is not None:
        notes.extend(n for n in affect.notes if n not in notes)

    return Translation(
        headline=headline,
        utterances=utterances,
        confidence=confidence,
        valence=valence,
        arousal=arousal,
        urgent=urgent,
        notes=notes,
        heard_voice=heard,
        saw_behaviour=saw,
    )


def translate_video(
    prediction,
    audio: AudioReading | None = None,
) -> Translation:
    """Convenience wrapper: translate a :class:`~dogsai.predict.VideoPrediction`."""
    from .audio import read_audio

    if audio is None:
        audio = read_audio(prediction.video)
    return translate(
        spans=prediction.spans,
        audio=audio,
        affect=prediction.affect(),
        duration=prediction.meta.duration,
    )
