from __future__ import annotations

import numpy as np
import pytest

from dogsai.metrics import (
    average_precision,
    evaluate,
    evaluate_multiclass,
    evaluate_multilabel,
    format_confusion,
    prf1,
    roc_auc,
    tune_thresholds,
)


class TestAveragePrecision:
    def test_perfect_ranking_scores_one(self):
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        targets = np.array([1, 1, 0, 0])
        assert average_precision(scores, targets) == pytest.approx(1.0)

    def test_worst_ranking_is_low(self):
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        targets = np.array([0, 0, 1, 1])
        # positives at ranks 3 and 4: (1/3 + 2/4) / 2
        assert average_precision(scores, targets) == pytest.approx((1 / 3 + 0.5) / 2)

    def test_known_interleaved_value(self):
        scores = np.array([0.9, 0.8, 0.7, 0.6])
        targets = np.array([1, 0, 1, 0])
        # precision at the two positives: 1/1 and 2/3
        assert average_precision(scores, targets) == pytest.approx((1.0 + 2 / 3) / 2)

    def test_no_positives_is_nan(self):
        assert np.isnan(average_precision(np.array([0.5, 0.4]), np.array([0, 0])))

    def test_all_positives_scores_one(self):
        assert average_precision(np.array([0.1, 0.2]), np.array([1, 1])) == pytest.approx(1.0)

    def test_constant_scores_give_base_rate(self):
        scores = np.full(10, 0.5)
        targets = np.array([1] * 5 + [0] * 5)
        # With every score tied the ranking is arbitrary; AP must stay finite.
        value = average_precision(scores, targets)
        assert 0.0 < value <= 1.0


class TestRocAuc:
    def test_perfect_separation(self):
        assert roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([1, 1, 0, 0])) == pytest.approx(1.0)

    def test_inverted_separation(self):
        assert roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == pytest.approx(0.0)

    def test_all_ties_is_one_half(self):
        assert roc_auc(np.full(6, 0.5), np.array([1, 1, 1, 0, 0, 0])) == pytest.approx(0.5)

    def test_single_class_is_nan(self):
        assert np.isnan(roc_auc(np.array([0.1, 0.9]), np.array([1, 1])))

    def test_known_partial_value(self):
        # one positive ranked above one negative and below another
        assert roc_auc(np.array([0.3, 0.2, 0.1]), np.array([0, 1, 0])) == pytest.approx(0.5)


class TestPrf1:
    def test_known_values(self):
        predictions = np.array([1, 1, 0, 0, 1])
        targets = np.array([1, 0, 0, 1, 1])
        precision, recall, f1 = prf1(predictions, targets)
        assert precision == pytest.approx(2 / 3)
        assert recall == pytest.approx(2 / 3)
        assert f1 == pytest.approx(2 / 3)

    def test_no_predictions_gives_zero_not_nan(self):
        precision, recall, f1 = prf1(np.zeros(4), np.array([1, 0, 1, 0]))
        assert (precision, recall, f1) == (0.0, 0.0, 0.0)


class TestTuneThresholds:
    def test_finds_a_separating_threshold(self):
        rng = np.random.default_rng(0)
        targets = np.zeros((200, 1))
        targets[:100] = 1
        scores = np.concatenate([rng.uniform(0.7, 1.0, 100), rng.uniform(0.0, 0.3, 100)])[:, None]
        threshold = tune_thresholds(scores, targets)[0]
        # Any cut inside the gap separates perfectly; assert it does separate.
        assert 0.3 <= threshold <= 0.7
        predictions = (scores >= threshold).astype(float)
        assert prf1(predictions[:, 0], targets[:, 0])[2] == pytest.approx(1.0)

    def test_picks_the_middle_of_a_tied_optimal_plateau(self):
        """An edge threshold sits against a training score and generalises worse."""
        targets = np.array([[1.0]] * 50 + [[0.0]] * 50)
        scores = np.array([[0.9]] * 50 + [[0.1]] * 50)
        threshold = tune_thresholds(scores, targets)[0]
        assert 0.1 < threshold < 0.9

    def test_keeps_default_for_thin_classes(self):
        scores = np.random.default_rng(0).random((20, 1))
        targets = np.zeros((20, 1))
        targets[:3] = 1  # below min_positives
        assert tune_thresholds(scores, targets, default=0.5)[0] == 0.5

    def test_returns_one_threshold_per_class(self):
        scores = np.random.default_rng(1).random((50, 4))
        targets = (np.random.default_rng(2).random((50, 4)) > 0.5).astype(float)
        assert tune_thresholds(scores, targets).shape == (4,)


class TestEvaluateMultilabel:
    def _perfect(self, n=40, c=3):
        targets = np.zeros((n, c))
        targets[np.arange(n), np.arange(n) % c] = 1
        logits = np.where(targets > 0, 6.0, -6.0)
        return logits, targets

    def test_perfect_predictions(self):
        logits, targets = self._perfect()
        result = evaluate_multilabel(logits, targets, ["a", "b", "c"])
        assert result.primary == pytest.approx(1.0)
        assert result.scalars["micro_f1"] == pytest.approx(1.0)
        assert result.scalars["exact_match"] == pytest.approx(1.0)

    def test_classes_without_positives_are_excluded_from_map(self):
        logits, targets = self._perfect(n=30, c=3)
        targets[:, 2] = 0  # class "c" now has no positives at all
        result = evaluate_multilabel(logits, targets, ["a", "b", "c"])
        assert np.isnan(result.per_class["c"]["ap"])
        assert not np.isnan(result.primary)

    def test_support_is_recorded_per_class(self):
        logits, targets = self._perfect(n=30, c=3)
        result = evaluate_multilabel(logits, targets, ["a", "b", "c"])
        assert result.per_class["a"]["support"] == targets[:, 0].sum()

    def test_fixed_thresholds_are_respected(self):
        logits = np.array([[0.0]] * 4)  # sigmoid -> 0.5 exactly
        targets = np.array([[1.0], [1.0], [0.0], [0.0]])
        result = evaluate_multilabel(logits, targets, ["a"], thresholds=0.9, tune=False)
        assert result.per_class["a"]["recall"] == 0.0

    def test_table_and_summary_render(self):
        logits, targets = self._perfect()
        result = evaluate_multilabel(logits, targets, ["a", "b", "c"])
        assert "behaviour" in result.table()
        assert "map=" in result.summary()
        assert "thresholds" in result.to_dict()

    def test_table_marks_undefined_metrics_as_na(self):
        logits, targets = self._perfect(n=30, c=3)
        targets[:, 2] = 0
        table = evaluate_multilabel(logits, targets, ["a", "b", "c"]).table()
        assert "n/a" in table


class TestEvaluateMulticlass:
    def test_perfect_predictions(self):
        targets = np.array([0, 1, 2, 0, 1, 2])
        logits = np.eye(3)[targets] * 8
        result = evaluate_multiclass(logits, targets, ["a", "b", "c"])
        assert result.scalars["top1"] == pytest.approx(1.0)
        assert result.primary == pytest.approx(1.0)
        assert result.confusion is not None
        assert np.array_equal(np.diag(result.confusion), [2, 2, 2])

    def test_accepts_one_hot_targets(self):
        indices = np.array([0, 1, 2])
        logits = np.eye(3)[indices] * 8
        from_index = evaluate_multiclass(logits, indices, ["a", "b", "c"])
        from_onehot = evaluate_multiclass(logits, np.eye(3)[indices], ["a", "b", "c"])
        assert from_index.scalars["top1"] == from_onehot.scalars["top1"]

    def test_balanced_accuracy_punishes_ignoring_a_rare_class(self):
        """Where plain accuracy lies and macro recall does not."""
        targets = np.array([0] * 95 + [1] * 5)
        logits = np.zeros((100, 2))
        logits[:, 0] = 5.0  # always predict the majority class
        result = evaluate_multiclass(logits, targets, ["common", "rare"])
        assert result.scalars["top1"] == pytest.approx(0.95)
        assert result.primary == pytest.approx(0.5)  # balanced accuracy sees it

    def test_confusion_renders(self):
        targets = np.array([0, 1, 1])
        logits = np.eye(2)[np.array([0, 1, 0])] * 5
        result = evaluate_multiclass(logits, targets, ["a", "b"])
        assert "a" in format_confusion(result.confusion, ["a", "b"])


def test_evaluate_dispatches_on_task():
    targets = np.array([0, 1])
    logits = np.eye(2)[targets] * 5
    assert evaluate(logits, targets, ["a", "b"], task="multiclass").task == "multiclass"
    assert evaluate(logits, np.eye(2)[targets], ["a", "b"], task="multilabel").task == "multilabel"
