from __future__ import annotations

import numpy as np
import pytest

from dogsai.labels import BehaviourSpan
from dogsai.predict import median_smooth, spans_from_mask


class TestMedianSmooth:
    def test_removes_a_single_window_spike(self):
        scores = np.array([[0.1], [0.1], [0.9], [0.1], [0.1]])
        smoothed = median_smooth(scores, 3)
        assert smoothed[2, 0] == pytest.approx(0.1)

    def test_preserves_a_sustained_run(self):
        scores = np.array([[0.1], [0.9], [0.9], [0.9], [0.1]])
        smoothed = median_smooth(scores, 3)
        assert smoothed[2, 0] == pytest.approx(0.9)

    def test_shape_is_unchanged(self):
        scores = np.random.default_rng(0).random((11, 4))
        assert median_smooth(scores, 5).shape == scores.shape

    def test_width_one_is_identity(self):
        scores = np.random.default_rng(0).random((6, 2))
        assert np.array_equal(median_smooth(scores, 1), scores)

    def test_even_width_is_promoted_to_odd(self):
        scores = np.random.default_rng(0).random((9, 1))
        assert median_smooth(scores, 4).shape == scores.shape

    def test_short_sequences_pass_through(self):
        scores = np.array([[0.5], [0.2]])
        assert np.array_equal(median_smooth(scores, 5), scores)

    def test_edges_are_not_dragged_to_zero(self):
        scores = np.full((7, 1), 0.8)
        assert median_smooth(scores, 3)[0, 0] == pytest.approx(0.8)


class TestSpansFromMask:
    def times(self, n, span=1.0, hop=1.0):
        return [(i * hop, i * hop + span) for i in range(n)]

    def test_one_contiguous_run_becomes_one_span(self):
        mask = np.array([False, True, True, True, False])
        scores = np.array([0.1, 0.8, 0.9, 0.7, 0.1])
        spans = spans_from_mask(mask, scores, self.times(5), "running")
        assert len(spans) == 1
        assert spans[0].start == 1.0 and spans[0].end == 4.0
        assert spans[0].peak == pytest.approx(0.9)
        assert spans[0].score == pytest.approx(0.8, abs=0.01)

    def test_two_runs_stay_separate_without_merging(self):
        mask = np.array([True, False, False, True])
        scores = np.array([0.9, 0.1, 0.1, 0.9])
        spans = spans_from_mask(mask, scores, self.times(4), "barking", merge_gap=0.0)
        assert len(spans) == 2

    def test_short_gap_is_bridged(self):
        mask = np.array([True, False, True])
        scores = np.array([0.9, 0.1, 0.9])
        spans = spans_from_mask(mask, scores, self.times(3), "barking", merge_gap=1.5)
        assert len(spans) == 1
        assert spans[0].start == 0.0 and spans[0].end == 3.0

    def test_large_gap_is_not_bridged(self):
        mask = np.array([True, False, False, False, True])
        scores = np.full(5, 0.9)
        spans = spans_from_mask(mask, scores, self.times(5), "barking", merge_gap=0.5)
        assert len(spans) == 2

    def test_short_spans_are_dropped(self):
        mask = np.array([True, False, True, True, True])
        scores = np.full(5, 0.9)
        spans = spans_from_mask(mask, scores, self.times(5), "digging", min_duration=1.5)
        assert len(spans) == 1
        assert spans[0].duration >= 1.5

    def test_run_reaching_the_final_window_is_closed(self):
        mask = np.array([False, True, True])
        scores = np.array([0.1, 0.8, 0.8])
        spans = spans_from_mask(mask, scores, self.times(3), "running")
        assert len(spans) == 1
        assert spans[0].end == 3.0

    def test_empty_mask_yields_nothing(self):
        spans = spans_from_mask(np.zeros(4, dtype=bool), np.zeros(4), self.times(4), "x")
        assert spans == []

    def test_all_true_yields_one_full_span(self):
        spans = spans_from_mask(np.ones(4, dtype=bool), np.full(4, 0.7), self.times(4), "x")
        assert len(spans) == 1
        assert spans[0].start == 0.0 and spans[0].end == 4.0

    def test_merged_score_is_duration_weighted(self):
        mask = np.array([True, True, False, True])
        scores = np.array([1.0, 1.0, 0.0, 0.0])
        spans = spans_from_mask(mask, scores, self.times(4), "x", merge_gap=1.5)
        assert len(spans) == 1
        # Two windows at 1.0 and one at 0.0 -> weighted mean above the plain mean.
        assert 0.5 < spans[0].score <= 1.0


class TestBehaviourSpan:
    def test_duration_and_dict(self):
        span = BehaviourSpan("running", 1.0, 3.5, 0.8, 0.95)
        assert span.duration == pytest.approx(2.5)
        payload = span.to_dict()
        assert payload["behaviour"] == "running"
        assert payload["duration"] == 2.5

    def test_negative_duration_clamps_to_zero(self):
        assert BehaviourSpan("x", 3.0, 1.0, 0.5).duration == 0.0

    def test_str_is_human_readable(self):
        assert "running" in str(BehaviourSpan("running", 0.0, 1.0, 0.9))
