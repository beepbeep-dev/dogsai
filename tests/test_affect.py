from __future__ import annotations

import pytest

from dogsai.affect import AFFECT_MAP, UNSEEN_SIGNALS, read_affect
from dogsai.labels import DEFAULT_BEHAVIOURS, BehaviourSpan


def span(behaviour: str, start: float, end: float, score: float = 0.9) -> BehaviourSpan:
    return BehaviourSpan(behaviour, start, end, score, score)


ALL = list(DEFAULT_BEHAVIOURS)


class TestAffectMap:
    def test_every_taxonomy_behaviour_has_a_mapping(self):
        """A behaviour with no mapping is silently invisible to the read."""
        missing = [b for b in DEFAULT_BEHAVIOURS if b not in AFFECT_MAP]
        assert missing == []

    def test_weights_and_ranges_are_sane(self):
        for name, contribution in AFFECT_MAP.items():
            assert -1.0 <= contribution.valence <= 1.0, name
            assert 0.0 <= contribution.arousal <= 1.0, name
            assert 0.0 < contribution.weight <= 2.0, name

    def test_every_mapping_explains_itself(self):
        for name, contribution in AFFECT_MAP.items():
            assert contribution.note, f"{name} has no note"

    def test_stress_signals_lean_negative(self):
        for name, contribution in AFFECT_MAP.items():
            if contribution.stress_signal:
                assert contribution.valence < 0, name

    def test_ambiguous_signals_are_weighted_below_clear_ones(self):
        """Tail wagging is ambiguous; play is not. The weights must reflect that."""
        assert AFFECT_MAP["tail_wagging"].weight < AFFECT_MAP["playing"].weight
        assert AFFECT_MAP["standing"].weight < AFFECT_MAP["playing"].weight


class TestReadAffect:
    def test_no_spans_says_so_rather_than_guessing(self):
        reading = read_affect([], duration=10.0, known_behaviours=ALL)
        assert reading.label == "not enough to say"
        assert reading.confidence == 0.0
        assert reading.coverage == 0.0
        assert not reading.needs_human_review

    def test_play_reads_positive_and_activated(self):
        reading = read_affect([span("playing", 0, 9)], 10.0, ALL)
        assert reading.valence > 0.5
        assert reading.arousal > 0.6
        assert "excited" in reading.label

    def test_resting_reads_positive_and_calm(self):
        reading = read_affect([span("lying_down", 0, 9)], 10.0, ALL)
        assert reading.arousal < 0.3
        assert reading.valence > 0.0
        assert reading.label in ("relaxed and content", "quiet and settled")

    def test_freeze_and_bark_reads_negative_and_flags_review(self):
        spans = [span("alert_freeze", 0, 4), span("barking", 2, 6)]
        reading = read_affect(spans, 8.0, ALL)
        assert reading.valence < -0.2
        assert reading.arousal > 0.6
        assert reading.needs_human_review
        assert set(reading.stress_signals) == {"alert_freeze", "barking"}

    def test_two_stress_signals_trigger_review_even_if_valence_is_mild(self):
        spans = [span("yawning", 0, 1), span("scratching", 2, 3), span("playing", 3, 9)]
        reading = read_affect(spans, 10.0, ALL)
        assert len(reading.stress_signals) >= 2
        assert reading.needs_human_review

    def test_low_scoring_spans_are_ignored(self):
        reading = read_affect([span("playing", 0, 9, score=0.1)], 10.0, ALL, min_score=0.35)
        assert reading.confidence == 0.0
        assert reading.label == "not enough to say"

    def test_longer_spans_dominate_shorter_ones(self):
        mostly_play = read_affect(
            [span("playing", 0, 9), span("alert_freeze", 9, 9.3)], 10.0, ALL
        )
        mostly_freeze = read_affect(
            [span("playing", 0, 0.3), span("alert_freeze", 1, 9.5)], 10.0, ALL
        )
        assert mostly_play.valence > mostly_freeze.valence

    def test_score_weighting_favours_confident_detections(self):
        confident = read_affect([span("playing", 0, 5, 0.95), span("barking", 0, 5, 0.4)], 10, ALL)
        doubtful = read_affect([span("playing", 0, 5, 0.4), span("barking", 0, 5, 0.95)], 10, ALL)
        assert confident.valence > doubtful.valence

    def test_coverage_tracks_how_much_of_the_video_was_explained(self):
        sparse = read_affect([span("playing", 0, 1)], 20.0, ALL)
        dense = read_affect([span("playing", 0, 19)], 20.0, ALL)
        assert sparse.coverage < 0.2 < dense.coverage
        assert sparse.confidence < dense.confidence

    def test_thin_coverage_refuses_to_name_a_quadrant(self):
        reading = read_affect([span("playing", 0, 0.5)], 30.0, ALL)
        assert reading.label == "not enough to say"

    def test_narrow_taxonomy_lowers_confidence_and_says_why(self):
        """A 5-class model must not sound as certain as an 18-class one."""
        spans = [span("eating_drinking", 0, 9)]
        narrow = read_affect(spans, 10.0, ["eating_drinking", "yawning", "eliminating"])
        wide = read_affect(spans, 10.0, ALL)
        assert narrow.confidence < wide.confidence
        assert any("only recognises" in n for n in narrow.notes)

    def test_single_behaviour_is_marked_as_weak_evidence(self):
        reading = read_affect([span("eating_drinking", 0, 9)], 10.0, ALL)
        assert any("single signal" in n for n in reading.notes)

    def test_unmapped_detected_behaviours_are_disclosed(self):
        reading = read_affect(
            [span("playing", 0, 9)], 10.0, known_behaviours=["playing", "wearing_a_hat"]
        )
        assert any("no affect mapping" in n for n in reading.notes)

    def test_unknown_behaviours_in_spans_are_skipped_safely(self):
        reading = read_affect(
            [span("playing", 0, 5), span("not_a_behaviour", 5, 9)], 10.0, ALL
        )
        assert [name for name, _, _ in reading.evidence] == ["playing"]

    def test_confidence_never_reaches_certainty(self):
        reading = read_affect([span("playing", 0, 20)], 20.0, ALL)
        assert reading.confidence <= 1.0

    def test_evidence_is_sorted_by_duration(self):
        spans = [span("yawning", 0, 1), span("playing", 1, 8)]
        reading = read_affect(spans, 10.0, ALL)
        durations = [seconds for _, seconds, _ in reading.evidence]
        assert durations == sorted(durations, reverse=True)


class TestOutputHonesty:
    def test_dict_output_carries_the_disclaimer(self):
        payload = read_affect([span("playing", 0, 9)], 10.0, ALL).to_dict()
        assert "disclaimer" in payload
        assert "not a measurement of emotion" in payload["disclaimer"]
        assert payload["not_observable_by_this_model"]

    def test_rendered_output_states_limits_and_shows_evidence(self):
        text = read_affect([span("playing", 0, 9)], 10.0, ALL).render()
        assert "valence" in text and "arousal" in text
        assert "confidence" in text
        assert "cannot see" in text
        assert "behaviourist" in text
        assert "playing" in text

    def test_unseen_signal_list_is_specific(self):
        assert any("tail position" in s for s in UNSEEN_SIGNALS)
        assert any("audio" in s for s in UNSEEN_SIGNALS)

    @pytest.mark.parametrize("behaviour", list(DEFAULT_BEHAVIOURS))
    def test_every_behaviour_produces_a_finite_reading(self, behaviour):
        reading = read_affect([span(behaviour, 0, 9)], 10.0, ALL)
        assert -1.0 <= reading.valence <= 1.0
        assert 0.0 <= reading.arousal <= 1.0
        assert 0.0 <= reading.confidence <= 1.0
        assert reading.label
