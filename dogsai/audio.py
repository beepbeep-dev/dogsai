"""Dog vocalisations: detecting them, typing them, and reading what they suggest.

Barks are not sentences
-----------------------
The honest headline first, because this is the module where it is easiest to
oversell: **no wording is decoded here, because there is none to decode.** Dog
vocalisations are not language. They carry no words, no syntax and no propositional
content, so there is no sentence hiding inside a bark. A bark is closer to a laugh,
a gasp or a scream than to a word: an involuntary signal shaped by internal state,
not a symbol chosen to mean something.

That does not make it uninformative, and :mod:`dogsai.translate` renders what is
found here into plain first-person English. The distinction that matters is that
such output carries *meaning* across from one signalling system to another; it is
not the recovery of a sentence the dog composed.

What acoustics genuinely supports is narrower and still useful. Dog bark structure
varies systematically with context, and the mapping is well replicated (Pongrácz,
Molnár & Miklósi and others on the acoustic structure of dog barks):

* **Pitch.** Low-pitched vocalisations skew agonistic — threat, warning, guarding.
  High-pitched ones skew towards fear, distress, play or greeting.
* **Tonality.** Noisy, atonal barks skew agonistic; tonal, harmonic ones skew
  friendly, playful, or distressed-but-not-threatening.
* **Rate.** Rapid barks with short gaps mean high arousal and usually alarm;
  isolated barks with long gaps mean lower arousal, often curiosity.

So this module reports **vocalisation type, arousal, and a likely context** — the
same two axes as :mod:`dogsai.affect`, from a different sensor. That is a
meaningful thing to know about a clip, and this module's own output stays in those
terms.

Two further limits, stated plainly:

1. **The typing here is rule-based DSP, not a trained classifier.** The features
   (pitch, tonality, duration, rate) are computed from scratch with numpy and
   compared against thresholds drawn from the literature. No labelled bark dataset
   was used to fit them, so treat the type as a well-motivated guess. A learned
   classifier on annotated audio would be better, and slots in behind the same
   interface.
2. **Anything can make a noise.** The event detector finds *sounds*, and cannot
   know whether a given one came from your dog, the television, a car, or you. It
   reports what it hears with a confidence, and confidence drops when a sound does
   not look much like a dog.

If a vocalisation worries you — particularly a yelp, which suggests pain — that is
a vet question, and this module's output is not a substitute for one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Dog vocal range. Barks sit roughly 200-900 Hz fundamental; whines and yelps
# reach much higher. The window is deliberately generous at the top so whines are
# not silently clipped into the bark range.
F0_MIN = 80.0
F0_MAX = 2500.0

# Event kinds that are explicitly not attributed to the dog.
NOT_THE_DOG: frozenset[str] = frozenset({"sound", "speech"})

DEFAULT_SR = 16000
FRAME = 512      # 32 ms at 16 kHz
HOP = 160        # 10 ms


class AudioError(RuntimeError):
    """Raised when a file has no usable audio."""


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def has_audio(path: str | Path) -> bool:
    import av

    try:
        with av.open(str(path)) as container:
            return bool(container.streams.audio)
    except Exception:
        return False


def load_audio(path: str | Path, sample_rate: int = DEFAULT_SR) -> tuple[np.ndarray, int]:
    """Decode the audio track to mono float32 in [-1, 1].

    Resampling and downmixing are done by ffmpeg through PyAV rather than by hand:
    it handles the planar/packed and integer/float format matrix that AAC, MP3 and
    PCM sources arrive in, which is a large amount of fiddly code to get wrong.
    """
    import av

    path = str(path)
    try:
        with av.open(path) as container:
            if not container.streams.audio:
                raise AudioError(f"{path} has no audio stream")
            stream = container.streams.audio[0]
            resampler = av.audio.resampler.AudioResampler(
                format="flt", layout="mono", rate=sample_rate
            )
            chunks: list[np.ndarray] = []
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(-1))
            # Flush the resampler's internal buffer, or the tail is lost.
            for resampled in resampler.resample(None):
                chunks.append(resampled.to_ndarray().reshape(-1))
    except AudioError:
        raise
    except Exception as exc:
        raise AudioError(f"failed to decode audio from {path}: {exc}") from exc

    if not chunks:
        raise AudioError(f"decoded no audio samples from {path}")
    audio = np.concatenate(chunks).astype(np.float32)
    peak = float(np.abs(audio).max())
    if peak > 1.0:
        audio /= peak
    return audio, sample_rate


# ---------------------------------------------------------------------------
# signal features, written from scratch on numpy
# ---------------------------------------------------------------------------
def frame_signal(audio: np.ndarray, frame: int = FRAME, hop: int = HOP) -> np.ndarray:
    """Split into overlapping frames as a strided view (no copy)."""
    if len(audio) < frame:
        audio = np.pad(audio, (0, frame - len(audio)))
    n = 1 + (len(audio) - frame) // hop
    return np.lib.stride_tricks.as_strided(
        audio, shape=(n, frame), strides=(audio.strides[0] * hop, audio.strides[0])
    )


def magnitude_spectrogram(frames: np.ndarray) -> np.ndarray:
    window = np.hanning(frames.shape[1]).astype(np.float32)
    return np.abs(np.fft.rfft(frames * window, axis=1))


def rms_db(frames: np.ndarray) -> np.ndarray:
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-10))


def spectral_centroid(spectrum: np.ndarray, sample_rate: int) -> np.ndarray:
    """Centre of spectral mass in Hz — a brightness measure."""
    freqs = np.fft.rfftfreq((spectrum.shape[1] - 1) * 2, 1.0 / sample_rate)
    total = np.maximum(spectrum.sum(axis=1), 1e-12)
    return (spectrum * freqs).sum(axis=1) / total


def spectral_flatness(spectrum: np.ndarray) -> np.ndarray:
    """Geometric over arithmetic mean: ~0 for a pure tone, ~1 for white noise.

    This is the tonality axis the bark literature keys on. Growls and threat barks
    are noisy (high flatness); whines and howls are tonal (low flatness).
    """
    power = np.maximum(spectrum, 1e-12) ** 2
    geometric = np.exp(np.log(power).mean(axis=1))
    arithmetic = power.mean(axis=1)
    return geometric / np.maximum(arithmetic, 1e-12)


def spectral_rolloff(spectrum: np.ndarray, sample_rate: int, fraction: float = 0.95) -> np.ndarray:
    freqs = np.fft.rfftfreq((spectrum.shape[1] - 1) * 2, 1.0 / sample_rate)
    cumulative = np.cumsum(spectrum, axis=1)
    totals = np.maximum(cumulative[:, -1:], 1e-12)
    index = np.argmax(cumulative >= fraction * totals, axis=1)
    return freqs[index]


def estimate_f0(
    segment: np.ndarray,
    sample_rate: int,
    f0_min: float = F0_MIN,
    f0_max: float = F0_MAX,
) -> tuple[float, float]:
    """Fundamental frequency and periodicity, via normalised autocorrelation.

    Returns ``(f0_hz, periodicity)`` where periodicity in [0, 1] is the height of
    the autocorrelation peak — effectively "how convincingly pitched is this".
    Autocorrelation is chosen over a plain spectral peak because a bark's loudest
    spectral component is frequently a harmonic rather than the fundamental, which
    makes naive peak-picking report double the true pitch.
    """
    segment = segment.astype(np.float64)
    segment = segment - segment.mean()
    if len(segment) < 64 or not np.any(segment):
        return 0.0, 0.0

    # FFT-based autocorrelation, zero-padded to avoid circular wraparound.
    size = 1 << int(math.ceil(math.log2(2 * len(segment))))
    spectrum = np.fft.rfft(segment, size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum))[: len(segment)]
    if correlation[0] <= 0:
        return 0.0, 0.0
    correlation /= correlation[0]

    min_lag = max(2, int(sample_rate / f0_max))
    max_lag = min(len(correlation) - 2, int(sample_rate / f0_min))
    if max_lag <= min_lag:
        return 0.0, 0.0
    window = correlation[min_lag : max_lag + 1]
    best = float(window.max())
    if best <= 0.15:  # nothing convincingly periodic
        return 0.0, max(0.0, best)

    # Octave errors are the central difficulty of autocorrelation pitch tracking,
    # and they cut both ways:
    #
    #  * A signal periodic at T is also periodic at 2T and 3T, and those peaks can
    #    be *taller* than the true one, because the true period rarely falls on a
    #    whole number of samples while some multiple of it does. (16 kHz / 1200 Hz
    #    = 13.33 samples, but 3x that is exactly 40 — so a 1200 Hz tone gets
    #    reported as 400 Hz.) That argues for preferring the shortest lag.
    #  * But when a real low fundamental is present with a louder harmonic above
    #    it, the shortest lag is the harmonic, and preferring it reports 600 Hz for
    #    a signal whose fundamental genuinely is 300 Hz.
    #
    # Neither preference is right on its own. What separates the two cases is the
    # spectrum: a genuine fundamental has audible energy at its own frequency,
    # while a spurious sub-harmonic has none. So gather every candidate peak within
    # tolerance and take the lowest frequency that the spectrum actually supports.
    tolerance = 0.85 * best
    candidates = [
        lag
        for lag in range(min_lag + 1, max_lag)
        if correlation[lag] >= tolerance
        and correlation[lag] >= correlation[lag - 1]
        and correlation[lag] >= correlation[lag + 1]
    ]
    if not candidates:
        candidates = [int(np.argmax(window)) + min_lag]

    spectrum_mag = np.abs(np.fft.rfft(segment * np.hanning(len(segment))))
    freqs = np.fft.rfftfreq(len(segment), 1.0 / sample_rate)
    reference = float(spectrum_mag.max()) + 1e-12

    def supported(lag: int) -> bool:
        """Is there real spectral energy at the frequency this lag implies?"""
        freq = sample_rate / lag
        band = (freqs >= freq * 0.85) & (freqs <= freq * 1.15)
        return band.any() and float(spectrum_mag[band].max()) / reference > 0.1

    # Longest lag first = lowest frequency first.
    peak = candidates[0]
    for lag in sorted(candidates, reverse=True):
        if supported(lag):
            peak = lag
            break
    value = float(correlation[peak])

    # Parabolic interpolation around the peak for sub-sample lag resolution.
    if 0 < peak < len(correlation) - 1:
        a, b, c = correlation[peak - 1], correlation[peak], correlation[peak + 1]
        denominator = 2 * (2 * b - a - c)
        if abs(denominator) > 1e-12:
            peak = peak + (c - a) / denominator
    return float(sample_rate / peak), value


# ---------------------------------------------------------------------------
# event detection
# ---------------------------------------------------------------------------
@dataclass
class VocalEvent:
    """One detected sound, typed and interpreted."""

    start: float
    end: float
    kind: str
    f0: float
    periodicity: float
    flatness: float
    centroid: float
    loudness_db: float
    confidence: float
    meaning: str = ""
    valence: float = 0.0
    arousal: float = 0.0
    concerning: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "kind": self.kind,
            "likely_context": self.meaning,
            "pitch_hz": round(self.f0, 1),
            "tonality": round(1.0 - self.flatness, 3),
            "loudness_db": round(self.loudness_db, 1),
            "confidence": round(self.confidence, 3),
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "concerning": self.concerning,
        }

    def __str__(self) -> str:
        pitch = f"{self.f0:4.0f}Hz" if self.f0 > 0 else "  --  "
        return (
            f"{self.start:6.2f}s-{self.end:6.2f}s  {self.kind:<10} "
            f"{pitch}  conf {self.confidence:.2f}  {self.meaning}"
        )


def detect_events(
    audio: np.ndarray,
    sample_rate: int,
    min_duration: float = 0.045,
    max_duration: float = 4.0,
    merge_gap: float = 0.06,
    open_db: float = 9.0,
    close_db: float = 5.0,
) -> list[tuple[float, float]]:
    """Find sound events by energy, relative to the clip's own noise floor.

    The thresholds are offsets above a low percentile of frame energy rather than
    absolute levels, because recording gain varies wildly between phones, and any
    fixed dB threshold is wrong for most files. Hysteresis (open high, close low)
    stops a single bark from being chopped into three by amplitude wobble.
    """
    frames = frame_signal(audio)
    energy = rms_db(frames)
    if len(energy) == 0:
        return []
    floor = float(np.percentile(energy, 20))
    span = float(np.percentile(energy, 99)) - floor
    if span < 3.0:
        return []  # essentially silence or steady noise: nothing to segment

    # Scale the thresholds when the clip has little dynamic range, so a quiet
    # recording still yields events instead of nothing.
    scale = min(1.0, span / 25.0)
    open_level = floor + open_db * scale
    close_level = floor + close_db * scale

    seconds_per_frame = HOP / sample_rate
    events: list[tuple[float, float]] = []
    start: int | None = None
    for i, level in enumerate(energy):
        if start is None and level >= open_level:
            start = i
        elif start is not None and level < close_level:
            events.append((start * seconds_per_frame, i * seconds_per_frame))
            start = None
    if start is not None:
        events.append((start * seconds_per_frame, len(energy) * seconds_per_frame))

    merged: list[tuple[float, float]] = []
    for begin, finish in events:
        if merged and begin - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], finish)
        else:
            merged.append((begin, finish))
    return [
        (a, min(b, a + max_duration))
        for a, b in merged
        if b - a >= min_duration
    ]


# ---------------------------------------------------------------------------
# typing and interpretation
# ---------------------------------------------------------------------------
# valence, arousal, plain-English context, whether it warrants attention.
_INTERPRETATION: dict[str, tuple[float, float, str, bool]] = {
    "yelp": (-0.85, 0.9,
             "a sudden sharp cry — most often pain or being startled; worth checking",
             True),
    "growl": (-0.65, 0.55,
              "a warning: distance-increasing, often guarding something or "
              "uncomfortable with what is approaching (also occurs in play)",
              True),
    "whine": (-0.4, 0.5,
              "wanting something or unsettled — attention, access, or relief; "
              "also an appeasement signal",
              False),
    "howl": (-0.25, 0.5,
             "a long-distance contact call, typically about being alone or "
             "answering another sound",
             False),
    "bark_alarm": (-0.4, 0.8,
                   "low and harsh, repeated: alarm or threat — something has been "
                   "noticed and is not welcome",
                   True),
    "bark_excited": (0.3, 0.8,
                     "high and clear: excitement, greeting or play — the friendly "
                     "end of barking",
                     False),
    "bark": (-0.05, 0.7,
             "barking, without a clear lean either way from its acoustics alone",
             False),
    "pant": (0.0, 0.4,
             "panting — usually cooling down, but also a stress response; the "
             "difference is in the context, not the sound",
             False),
    "sound": (0.0, 0.3,
              "a sound that does not clearly match a known dog vocalisation — it "
              "may not be the dog at all",
              False),
    "speech": (0.0, 0.2,
               "this one sounds like a person talking, not the dog",
               False),
}


def speech_likeness(
    spectrum: np.ndarray,
    sample_rate: int,
    f0: float,
    periodicity: float,
    duration: float,
    modulation: float,
) -> float:
    """How much an event looks like human speech rather than a dog.

    Worth its own function because it is the dominant false positive in real
    footage: people talk to their dogs while filming them, and speech occupies the
    same acoustic region as a howl or a growl — sustained, tonal, pitch-modulated,
    fundamental between roughly 85 and 300 Hz.

    The discriminators used here are the ones that survive a phone microphone:
    speech has its fundamental in that narrow band, concentrates most of its
    energy in the 300-3400 Hz formant region, is continuously pitch-modulated by
    prosody, and lasts longer than a bark. None of these is decisive alone, so the
    score is additive and the caller applies a threshold.
    """
    if duration < 0.25 or f0 <= 0:
        return 0.0
    freqs = np.fft.rfftfreq((spectrum.shape[1] - 1) * 2, 1.0 / sample_rate)
    mean_spectrum = spectrum.mean(axis=0)
    total = float(mean_spectrum.sum()) + 1e-12
    formant_band = float(mean_spectrum[(freqs >= 300) & (freqs <= 3400)].sum()) / total
    high_band = float(mean_spectrum[freqs > 4000].sum()) / total

    score = 0.0
    if 85.0 <= f0 <= 300.0:
        score += 0.35
    if formant_band > 0.6:
        score += 0.2
    if 0.05 < modulation < 0.45:
        score += 0.2       # prosody: varying, but not the wild sweeps of a howl
    if duration > 0.6:
        score += 0.15
    if periodicity > 0.45:
        score += 0.1
    if high_band > 0.25:
        score -= 0.2       # barks and yelps carry far more high-frequency energy
    return max(0.0, min(1.0, score))


def classify_event(
    audio: np.ndarray,
    sample_rate: int,
    start: float,
    end: float,
    bark_rate: float = 0.0,
) -> VocalEvent:
    """Type one event from its acoustics.

    Thresholds follow the published structure-to-context mapping: pitch splits
    threat from fear/play, tonality splits agonistic from friendly/distress, and
    duration separates barks from growls, whines and howls. Confidence reflects
    how cleanly the features fall inside a category rather than being asserted.
    """
    begin = max(0, int(start * sample_rate))
    finish = min(len(audio), int(end * sample_rate))
    segment = audio[begin:finish]
    if len(segment) < 32:
        return VocalEvent(start, end, "sound", 0.0, 0.0, 1.0, 0.0, -60.0, 0.1,
                          *_INTERPRETATION["sound"][2:3], 0.0, 0.3, False)

    frames = frame_signal(segment, frame=min(FRAME, len(segment)), hop=HOP)
    spectrum = magnitude_spectrogram(frames)
    flatness = float(np.median(spectral_flatness(spectrum)))
    centroid = float(np.median(spectral_centroid(spectrum, sample_rate)))
    loudness = float(np.max(rms_db(frames)))
    f0, periodicity = estimate_f0(segment, sample_rate)
    duration = end - start
    tonal = periodicity > 0.35 and flatness < 0.30

    # Pitch contour, for separating a modulated howl from a steady whine.
    pitches = []
    step = max(1, len(segment) // 6)
    for offset in range(0, max(1, len(segment) - step), step):
        pitch, clarity = estimate_f0(segment[offset : offset + step], sample_rate)
        if pitch > 0 and clarity > 0.3:
            pitches.append(pitch)
    modulation = (float(np.std(pitches)) / max(1.0, float(np.mean(pitches)))) if len(pitches) > 2 else 0.0

    speech = speech_likeness(spectrum, sample_rate, f0, periodicity, duration, modulation)

    kind = "sound"
    confidence = 0.35
    if speech > 0.6:
        # Almost every home dog video has a human talking in it, and speech is
        # sustained, tonal and pitch-modulated in the same 100-300 Hz band as a
        # howl or growl. Without this guard the commonest sound in the clip gets
        # confidently mistyped as the dog, which is worse than saying nothing.
        valence, arousal, meaning, concerning = _INTERPRETATION["speech"]
        return VocalEvent(
            start=start, end=end, kind="speech", f0=f0, periodicity=periodicity,
            flatness=flatness, centroid=centroid, loudness_db=loudness,
            confidence=min(0.6, speech), meaning=meaning,
            valence=valence, arousal=arousal, concerning=concerning,
        )

    # Ordered most-specific first. Each branch keys on the discriminators the
    # literature actually supports — duration, pitch band, and how cleanly
    # periodic the signal is — rather than on a single composite "tonal" flag,
    # which conflated a rough low growl with a clean low howl.
    if duration < 0.18 and f0 > 600 and periodicity > 0.4:
        kind, confidence = "yelp", 0.55
    elif duration >= 0.3 and 0 < f0 < 350 and (flatness > 0.1 or periodicity < 0.65):
        kind, confidence = "growl", 0.55
    elif 0.7 <= duration <= 3.5 and 200 <= f0 <= 1400 and periodicity > 0.5 and modulation > 0.12:
        kind, confidence = "howl", 0.5
    elif duration >= 0.22 and f0 > 450 and periodicity > 0.35:
        kind, confidence = "whine", 0.55
    elif 0.05 <= duration <= 0.6 and loudness > -40 and centroid > 400:
        # A bark: split on the two axes the literature says carry the meaning.
        if f0 > 0 and f0 < 450 and flatness > 0.14:
            kind, confidence = "bark_alarm", 0.5
        elif f0 >= 450 and tonal:
            kind, confidence = "bark_excited", 0.5
        else:
            kind, confidence = "bark", 0.45
    elif duration > 0.6 and flatness > 0.35 and centroid < 2500 and not tonal:
        kind, confidence = "pant", 0.4

    # Rapid repetition is itself evidence of alarm rather than greeting.
    if kind in ("bark", "bark_excited") and bark_rate >= 3.0 and f0 < 700:
        kind, confidence = "bark_alarm", min(0.55, confidence + 0.05)

    valence, arousal, meaning, concerning = _INTERPRETATION[kind]
    # Faster sequences read as more aroused.
    if bark_rate >= 3.0 and kind.startswith("bark"):
        arousal = min(1.0, arousal + 0.1)
    if loudness < -45:
        confidence *= 0.7  # very quiet: could be anything, or not the dog
    return VocalEvent(
        start=start, end=end, kind=kind, f0=f0, periodicity=periodicity,
        flatness=flatness, centroid=centroid, loudness_db=loudness,
        confidence=min(1.0, confidence), meaning=meaning,
        valence=valence, arousal=arousal, concerning=concerning,
    )


@dataclass
class AudioReading:
    """Everything heard in one clip."""

    duration: float
    events: list[VocalEvent] = field(default_factory=list)
    valence: float = 0.0
    arousal: float = 0.0
    vocal_fraction: float = 0.0
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def concerning(self) -> list[VocalEvent]:
        return [e for e in self.events if e.concerning]

    @property
    def dog_events(self) -> list[VocalEvent]:
        """Only the events attributed to the dog."""
        return [e for e in self.events if e.kind not in NOT_THE_DOG]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for event in self.events:
            out[event.kind] = out.get(event.kind, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def to_dict(self) -> dict:
        return {
            "duration": round(self.duration, 3),
            "n_events": len(self.events),
            "kinds": self.counts(),
            "valence": round(self.valence, 3),
            "arousal": round(self.arousal, 3),
            "vocal_fraction": round(self.vocal_fraction, 4),
            "confidence": round(self.confidence, 3),
            "events": [e.to_dict() for e in self.events],
            "notes": self.notes,
            "disclaimer": (
                "Dog vocalisations are not language and are not translated here. "
                "These are inferences about vocalisation type and likely context "
                "from acoustic structure (pitch, tonality, duration, rate), using "
                "rule-based signal processing rather than a trained classifier. "
                "The detector cannot know whether a given sound came from the dog."
            ),
        }

    def render(self) -> str:
        if not self.events:
            return (
                "audio: no vocalisations detected\n"
                "  (either the dog was quiet, or the clip has no usable audio)"
            )
        lines = [f"audio: {len(self.events)} vocalisation(s) detected", ""]
        for event in self.events:
            lines.append(f"  {event}")
        lines += [
            "",
            f"  summary: " + ", ".join(f"{k} x{v}" for k, v in self.counts().items()),
            f"  audio valence {self.valence:+.2f}   arousal {self.arousal:.2f}   "
            f"confidence {self.confidence:.0%}",
        ]
        if self.concerning:
            kinds = sorted({e.kind for e in self.concerning})
            lines += ["", f"  worth your attention: {', '.join(kinds)}"]
        if self.notes:
            lines += [""] + [f"  note: {n}" for n in self.notes]
        lines += [
            "",
            "  These are acoustic guesses at vocalisation type and likely context —",
            "  by rule, not by a trained model — and the detector cannot tell your",
            "  dog from the television. Barks contain no wording to decode; for these",
            "  signals rendered into plain English, see `dogsai translate`.",
        ]
        return "\n".join(lines)


def read_audio(
    path: str | Path,
    sample_rate: int = DEFAULT_SR,
    max_events: int = 200,
) -> AudioReading:
    """Detect, type and interpret every vocalisation in a video or audio file."""
    try:
        audio, sample_rate = load_audio(path, sample_rate)
    except AudioError as exc:
        return AudioReading(duration=0.0, notes=[str(exc)])

    duration = len(audio) / sample_rate
    spans = detect_events(audio, sample_rate)
    notes: list[str] = []
    if len(spans) > max_events:
        notes.append(
            f"{len(spans)} sound events found; only the {max_events} loudest were "
            f"analysed. A very noisy recording is hard to attribute to the dog."
        )
        spans = sorted(spans, key=lambda s: s[1] - s[0], reverse=True)[:max_events]
        spans.sort()

    # Local rate around each event, used as an arousal and alarm cue.
    onsets = np.array([s for s, _ in spans]) if spans else np.zeros(0)
    events: list[VocalEvent] = []
    for start, end in spans:
        if len(onsets) > 1:
            nearby = int(np.sum(np.abs(onsets - start) <= 1.0)) - 1
        else:
            nearby = 0
        events.append(classify_event(audio, sample_rate, start, end, bark_rate=float(nearby)))

    # Weight by confidence and duration. Events attributed to something other
    # than the dog — unclassifiable noise, and human speech — are excluded
    # entirely rather than merely down-weighted: they say nothing about the dog's
    # state, and letting them contribute would put a confident-looking number on
    # a clip where the only audible thing was a person talking.
    weights, valences, arousals = [], [], []
    for event in events:
        if event.kind in NOT_THE_DOG:
            continue
        weight = event.confidence * max(0.05, min(1.0, event.duration))
        weights.append(weight)
        valences.append(event.valence)
        arousals.append(event.arousal)

    if weights:
        total = float(sum(weights))
        valence = float(np.average(valences, weights=weights))
        arousal = float(np.average(arousals, weights=weights))
        confidence = min(1.0, total / 2.0) * min(1.0, 0.4 + 0.15 * len(weights))
    else:
        valence = arousal = confidence = 0.0
        if events:
            spoken = sum(1 for e in events if e.kind == "speech")
            notes.append(
                "sounds were detected but none matched a dog vocalisation clearly "
                "enough to interpret"
                + (f" ({spoken} sounded like a person talking)" if spoken else "")
            )

    vocal = sum(e.duration for e in events if e.kind not in NOT_THE_DOG)
    if len({e.kind for e in events if e.kind not in NOT_THE_DOG}) > 3:
        notes.append("several different vocalisation types — likely a busy or mixed scene")

    return AudioReading(
        duration=duration,
        events=events,
        valence=valence,
        arousal=arousal,
        vocal_fraction=min(1.0, vocal / max(duration, 1e-6)),
        confidence=confidence,
        notes=notes,
    )
