from __future__ import annotations

import pytest

from dogsai.advise import advise
from dogsai.affect import read_affect
from dogsai.audio import AudioReading, VocalEvent
from dogsai.labels import DEFAULT_BEHAVIOURS, BehaviourSpan
from dogsai.translate import _BEHAVIOUR_LINES, _VOICE_LINES, translate

ALL = list(DEFAULT_BEHAVIOURS)


def span(behaviour: str, start: float, end: float, score: float = 0.9) -> BehaviourSpan:
    return BehaviourSpan(behaviour, start, end, score, score)


def voice(kind: str, start: float, end: float, f0: float = 500.0,
          confidence: float = 0.6, concerning: bool = False) -> VocalEvent:
    from dogsai.audio import _INTERPRETATION

    valence, arousal, meaning, flag = _INTERPRETATION[kind]
    return VocalEvent(start, end, kind, f0, 0.6, 0.1, 1200, -20, confidence,
                      meaning, valence, arousal, concerning or flag)


def audio_of(*events: VocalEvent, duration: float = 10.0) -> AudioReading:
    dog = [e for e in events if e.kind not in ("sound", "speech")]
    if dog:
        valence = sum(e.valence for e in dog) / len(dog)
        arousal = sum(e.arousal for e in dog) / len(dog)
        confidence = 0.6
    else:
        valence = arousal = confidence = 0.0
    return AudioReading(duration=duration, events=list(events), valence=valence,
                        arousal=arousal, confidence=confidence)


class TestTranslate:
    def test_nothing_detected_says_so_instead_of_inventing_dialogue(self):
        result = translate(spans=[], audio=None, affect=None, duration=10.0)
        assert result.confidence == 0.0
        assert not result.utterances
        assert "not giving you much" in result.headline
        assert result.notes

    def test_behaviour_produces_a_grounded_utterance(self):
        spans = [span("playing", 0.0, 6.0)]
        result = translate(spans, affect=read_affect(spans, 10.0, ALL), duration=10.0)
        assert result.utterances
        line = result.utterances[0]
        assert line.text
        assert "playing" in line.basis   # traceable to a detection
        assert line.source == "behaviour"

    def test_every_utterance_traces_to_a_detection(self):
        spans = [span("playing", 0.0, 4.0), span("chewing", 5.0, 9.0)]
        result = translate(spans, audio=audio_of(voice("bark_excited", 1.0, 1.2)),
                           affect=read_affect(spans, 10.0, ALL), duration=10.0)
        assert result.utterances
        for utterance in result.utterances:
            assert utterance.basis
            assert utterance.source in ("behaviour", "voice")

    def test_voice_is_ordered_before_behaviour_at_the_same_moment(self):
        spans = [span("playing", 1.0, 6.0)]
        result = translate(spans, audio=audio_of(voice("growl", 1.0, 1.6)),
                           affect=read_affect(spans, 10.0, ALL), duration=10.0)
        at_one = [u for u in result.utterances if abs(u.start - 1.0) < 1e-6]
        assert at_one[0].source == "voice"

    def test_urgent_vocalisation_takes_over_the_headline(self):
        """A yelp during play must not be summarised as "having a lovely time"."""
        spans = [span("playing", 0.0, 9.0)]
        result = translate(spans, audio=audio_of(voice("yelp", 3.0, 3.1, f0=1400)),
                           affect=read_affect(spans, 10.0, ALL), duration=10.0)
        assert any(u.urgent for u in result.utterances)
        assert result.headline in _VOICE_LINES["yelp"]

    def test_conflicting_channels_are_reported_not_averaged(self):
        spans = [span("playing", 0.0, 9.0)]
        result = translate(spans, audio=audio_of(voice("yelp", 3.0, 3.1, f0=1400)),
                           affect=read_affect(spans, 10.0, ALL), duration=10.0)
        assert result.valence <= 0.15
        assert any("disagree" in n for n in result.notes)

    def test_repeated_behaviour_windows_are_collapsed(self):
        """Sliding windows fragment long behaviours; the text must not stutter."""
        spans = [span("playing", 0.0, 2.0, 0.9), span("playing", 2.1, 4.0, 0.9)]
        result = translate(spans, affect=read_affect(spans, 10.0, ALL), duration=10.0)
        texts = [u.text for u in result.utterances]
        assert len(texts) == len(set(texts))

    def test_very_short_spans_are_ignored(self):
        result = translate([span("playing", 0.0, 0.1)], duration=10.0, min_span=0.4)
        assert not any(u.source == "behaviour" for u in result.utterances)

    def test_silent_clip_is_noted(self):
        spans = [span("chewing", 0.0, 8.0)]
        result = translate(spans, audio=audio_of(duration=10.0),
                           affect=read_affect(spans, 10.0, ALL), duration=10.0)
        assert result.saw_behaviour and not result.heard_voice
        assert any("silent" in n for n in result.notes)

    def test_human_speech_is_disclosed_and_not_voiced_as_the_dog(self):
        speech = VocalEvent(1.0, 2.0, "speech", 150, 0.6, 0.1, 900, -20, 0.6, "person")
        result = translate([span("playing", 0.0, 8.0)], audio=audio_of(speech), duration=10.0)
        assert not result.heard_voice
        assert any("person talking" in n for n in result.notes)
        assert all(u.source == "behaviour" for u in result.utterances)

    def test_output_is_deterministic(self):
        spans = [span("playing", 0.0, 6.0)]
        a = translate(spans, affect=read_affect(spans, 10.0, ALL), duration=10.0)
        b = translate(spans, affect=read_affect(spans, 10.0, ALL), duration=10.0)
        assert [u.text for u in a.utterances] == [u.text for u in b.utterances]
        assert a.headline == b.headline

    def test_dict_and_render_state_what_this_is(self):
        spans = [span("playing", 0.0, 6.0)]
        result = translate(spans, affect=read_affect(spans, 10.0, ALL), duration=10.0)
        payload = result.to_dict()
        assert "do not use words" in payload["disclaimer"]
        text = result.render()
        assert "not speech decoded" in text
        assert "confidence" in text

    def test_every_taxonomy_behaviour_that_speaks_has_lines(self):
        for behaviour, options in _BEHAVIOUR_LINES.items():
            assert behaviour in DEFAULT_BEHAVIOURS, behaviour
            assert options and all(o.strip() for o in options)

    @pytest.mark.parametrize("behaviour", list(_BEHAVIOUR_LINES))
    def test_each_behaviour_alone_produces_a_line(self, behaviour):
        spans = [span(behaviour, 0.0, 8.0)]
        result = translate(spans, affect=read_affect(spans, 10.0, ALL), duration=10.0)
        assert result.utterances


class TestAdvise:
    def test_nothing_detected_yields_nothing(self):
        result = advise(spans=[], audio=None, affect=None)
        assert not result.suggestions
        assert "nothing to suggest" in result.render()

    def test_growl_advice_warns_against_punishing_it(self):
        """The single most important piece of advice in the module."""
        result = advise(audio=audio_of(voice("growl", 1.0, 1.8)))
        text = " ".join(s.text for s in result.suggestions)
        assert "not punish" in text or "Do not punish" in text
        assert result.urgent

    def test_yelp_routes_to_a_vet(self):
        result = advise(audio=audio_of(voice("yelp", 1.0, 1.1, f0=1400)))
        assert result.veterinary
        assert result.urgent

    def test_alert_freeze_is_high_priority(self):
        spans = [span("alert_freeze", 0.0, 3.0)]
        result = advise(spans=spans, affect=read_affect(spans, 5.0, ALL))
        assert any(s.priority == 1 for s in result.suggestions)

    def test_suggestions_are_priority_ordered(self):
        spans = [span("playing", 0.0, 5.0), span("alert_freeze", 5.0, 8.0)]
        result = advise(spans=spans, audio=audio_of(voice("yelp", 1.0, 1.1, f0=1400)),
                        affect=read_affect(spans, 10.0, ALL))
        priorities = [s.priority for s in result.suggestions]
        assert priorities == sorted(priorities)

    def test_suggestions_are_deduplicated(self):
        spans = [span("chewing", 0.0, 3.0), span("chewing", 4.0, 8.0)]
        result = advise(spans=spans, affect=read_affect(spans, 10.0, ALL))
        texts = [s.text for s in result.suggestions]
        assert len(texts) == len(set(texts))

    def test_capped_so_the_list_stays_readable(self):
        spans = [span(b, i * 1.0, i * 1.0 + 0.9) for i, b in enumerate(_BEHAVIOUR_LINES)]
        result = advise(spans=spans, max_suggestions=4)
        assert len(result.suggestions) <= 4

    def test_short_spans_do_not_trigger_advice(self):
        result = advise(spans=[span("chewing", 0.0, 0.2)], min_duration=0.5)
        assert not result.suggestions

    def test_every_suggestion_explains_itself(self):
        spans = [span("yawning", 0.0, 4.0)]
        result = advise(spans=spans, audio=audio_of(voice("whine", 1.0, 1.5)),
                        affect=read_affect(spans, 6.0, ALL))
        for suggestion in result.suggestions:
            assert suggestion.because
            assert 1 <= suggestion.priority <= 3

    def test_contains_no_aversive_or_dosage_advice(self):
        """Scope guard: nothing punitive, nothing medical-prescriptive."""
        from dogsai.advise import _BEHAVIOUR_ADVICE, _VOICE_ADVICE

        banned = ("punish the dog", "correction", "alpha", "dominance", "shock",
                  "choke", "prong", "mg", "dose", "dosage", "medicate")
        for group in (_BEHAVIOUR_ADVICE, _VOICE_ADVICE):
            for suggestions in group.values():
                for suggestion in suggestions:
                    lowered = suggestion.text.lower()
                    for word in banned:
                        assert word not in lowered, f"{word!r} in {suggestion.text!r}"

    def test_dict_carries_the_disclaimer(self):
        result = advise(audio=audio_of(voice("whine", 1.0, 1.6)))
        payload = result.to_dict()
        assert "not veterinary advice" in payload["disclaimer"]
        assert payload["suggestions"]

    def test_render_states_its_limits(self):
        text = advise(audio=audio_of(voice("whine", 1.0, 1.6))).render()
        assert "not veterinary advice" in text
        assert "behaviourist" in text
