"""Evaluation metrics.

Accuracy is the wrong headline number for this problem and it is worth being
explicit about why:

* The taxonomy is long-tailed.  ``standing`` may be 30% of windows and ``digging``
  0.5%, so a model that never predicts ``digging`` still scores well on plain
  accuracy.  Macro-averaged metrics are reported alongside micro ones so that
  failure is visible.
* The task is multi-label, so "correct" is not binary per sample.  Mean average
  precision over classes is the primary metric — it is threshold-free, so it
  measures the ranking the model actually learned rather than an arbitrary 0.5
  cut.
* Thresholds should be *fitted*, not assumed.  :func:`tune_thresholds` picks a
  per-class operating point on validation data, which is usually worth several
  points of F1 over a global 0.5 and costs nothing at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _as_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array)


def average_precision(scores: np.ndarray, targets: np.ndarray) -> float:
    """Area under the precision-recall curve for one class.

    Uses the step-wise definition ``AP = sum_k (R_k - R_{k-1}) * P_k`` over the
    score-sorted ranking, which is the same estimator scikit-learn uses and does
    not interpolate (interpolated 11-point AP is optimistic on small sets).
    Returns NaN when the class has no positives — such classes are excluded from
    the mean rather than counted as 0, which would be an arbitrary penalty.
    """
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    n_positive = float(targets.sum())
    if n_positive == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")  # stable: ties keep input order
    sorted_targets = targets[order]
    tp = np.cumsum(sorted_targets)
    fp = np.cumsum(1.0 - sorted_targets)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / n_positive
    # Only ranks where recall increases contribute.
    mask = sorted_targets > 0
    return float(np.sum(precision[mask]) / n_positive) if mask.any() else float("nan")


def roc_auc(scores: np.ndarray, targets: np.ndarray) -> float:
    """AUC via the rank-sum (Mann-Whitney U) identity, with tie correction."""
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    n_pos = float(targets.sum())
    n_neg = float(len(targets) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0  # average rank for ties
        i = j + 1
    rank_sum = ranks[targets > 0].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def prf1(
    predictions: np.ndarray, targets: np.ndarray
) -> tuple[float, float, float]:
    tp = float(np.sum((predictions > 0) & (targets > 0)))
    fp = float(np.sum((predictions > 0) & (targets == 0)))
    fn = float(np.sum((predictions == 0) & (targets > 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def tune_thresholds(
    scores: np.ndarray,
    targets: np.ndarray,
    default: float = 0.5,
    min_positives: int = 8,
) -> np.ndarray:
    """Per-class threshold that maximises F1 on the given (validation) scores.

    Classes with too few positives to fit a threshold reliably keep ``default`` —
    fitting on three examples produces a threshold that is pure noise and will
    hurt on the test set.
    """
    scores = _as_numpy(scores)
    targets = _as_numpy(targets)
    n_classes = scores.shape[1]
    out = np.full(n_classes, default, dtype=np.float64)
    for c in range(n_classes):
        y = targets[:, c]
        if y.sum() < min_positives:
            continue
        s = scores[:, c]
        # Candidates are the midpoints *between* observed scores, not the scores
        # themselves: a threshold equal to an observed score is knife-edge, and
        # which side a near-identical new score falls on is then a coin flip.
        observed = np.unique(np.round(s, 6))
        if len(observed) > 256:
            observed = np.unique(np.quantile(s, np.linspace(0.0, 1.0, 256)))
        edges = np.concatenate(([observed[0] - 1e-3], observed, [observed[-1] + 1e-3]))
        candidates = (edges[:-1] + edges[1:]) / 2.0
        f1s = np.array([prf1((s >= t).astype(np.float64), y)[2] for t in candidates])
        best = f1s.max()
        if best <= 0:
            continue
        # When a range of thresholds ties for the best F1 (common on
        # well-separated classes), take the middle of that plateau rather than
        # its edge: an edge threshold sits right against a training score and is
        # the first thing to break on new data.
        plateau = candidates[f1s >= best - 1e-9]
        out[c] = float(np.median(plateau))
    return out


@dataclass
class EvalResult:
    """Everything an evaluation pass produces, in one printable object."""

    task: str
    names: list[str]
    n_samples: int
    primary: float = 0.0
    primary_name: str = "map"
    scalars: dict[str, float] = field(default_factory=dict)
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: np.ndarray | None = None
    thresholds: np.ndarray | None = None

    def to_dict(self) -> dict:
        out = {
            "task": self.task,
            "n_samples": self.n_samples,
            "primary": self.primary,
            "primary_name": self.primary_name,
            **{k: float(v) for k, v in self.scalars.items()},
            "per_class": {
                k: {m: float(v) for m, v in stats.items()}
                for k, stats in self.per_class.items()
            },
        }
        if self.thresholds is not None:
            out["thresholds"] = [float(t) for t in self.thresholds]
        return out

    def table(self, sort_by: str | None = None) -> str:
        """Per-class breakdown as a fixed-width table."""
        if not self.per_class:
            return "(no per-class metrics)"
        columns = list(next(iter(self.per_class.values())).keys())
        rows = list(self.per_class.items())
        if sort_by and sort_by in columns:
            rows.sort(key=lambda kv: kv[1][sort_by])
        width = max(max(len(n) for n in self.per_class) + 1, len("behaviour") + 1)
        cell = 11
        header = "behaviour".ljust(width) + "".join(c.rjust(cell) for c in columns)
        lines = [header, "-" * len(header)]
        for name, stats in rows:
            cells = ""
            for column in columns:
                value = stats[column]
                if np.isnan(value):
                    cells += "n/a".rjust(cell)
                elif column == "support":
                    cells += f"{int(value)}".rjust(cell)
                else:
                    cells += f"{value:{cell}.4f}"
            lines.append(name.ljust(width) + cells)
        return "\n".join(lines)

    def summary(self) -> str:
        parts = [f"{self.primary_name}={self.primary:.4f}"]
        parts += [f"{k}={v:.4f}" for k, v in self.scalars.items()]
        return "  ".join(parts)


def evaluate_multilabel(
    logits: np.ndarray,
    targets: np.ndarray,
    names: list[str],
    thresholds: np.ndarray | float = 0.5,
    tune: bool = True,
) -> EvalResult:
    logits = _as_numpy(logits).astype(np.float64)
    targets = _as_numpy(targets).astype(np.float64)
    scores = 1.0 / (1.0 + np.exp(-logits))

    if tune:
        thresholds = tune_thresholds(scores, targets)
    elif np.isscalar(thresholds):
        thresholds = np.full(scores.shape[1], float(thresholds))
    thresholds = np.asarray(thresholds, dtype=np.float64)
    predictions = (scores >= thresholds[None, :]).astype(np.float64)

    per_class: dict[str, dict[str, float]] = {}
    aps, f1s = [], []
    for c, name in enumerate(names):
        ap = average_precision(scores[:, c], targets[:, c])
        auc = roc_auc(scores[:, c], targets[:, c])
        precision, recall, f1 = prf1(predictions[:, c], targets[:, c])
        support = float(targets[:, c].sum())
        per_class[name] = {
            "ap": ap,
            "auc": auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "thresh": float(thresholds[c]),
        }
        if not np.isnan(ap):
            aps.append(ap)
            f1s.append(f1)

    micro_p, micro_r, micro_f1 = prf1(predictions.ravel(), targets.ravel())
    # Exact-match ("subset") accuracy: strict, but the honest number for
    # "did we get the whole label set right".
    exact = float(np.mean(np.all(predictions == targets, axis=1)))
    mean_ap = float(np.mean(aps)) if aps else float("nan")

    return EvalResult(
        task="multilabel",
        names=list(names),
        n_samples=int(len(targets)),
        primary=mean_ap,
        primary_name="map",
        scalars={
            "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
            "micro_f1": micro_f1,
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "exact_match": exact,
        },
        per_class=per_class,
        thresholds=thresholds,
    )


def evaluate_multiclass(
    logits: np.ndarray, targets: np.ndarray, names: list[str]
) -> EvalResult:
    logits = _as_numpy(logits).astype(np.float64)
    targets = _as_numpy(targets)
    if targets.ndim == 2:
        truth = targets.argmax(axis=1)
    else:
        truth = targets.astype(np.int64)
    predictions = logits.argmax(axis=1)
    n = len(names)

    confusion = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(truth, predictions):
        confusion[t, p] += 1

    top1 = float(np.mean(predictions == truth))
    order = np.argsort(-logits, axis=1)
    k = min(5, n)
    topk = float(np.mean([truth[i] in order[i, :k] for i in range(len(truth))]))

    scores = np.exp(logits - logits.max(axis=1, keepdims=True))
    scores /= scores.sum(axis=1, keepdims=True)
    onehot = np.zeros_like(scores)
    onehot[np.arange(len(truth)), truth] = 1.0

    per_class: dict[str, dict[str, float]] = {}
    recalls, f1s, aps = [], [], []
    for c, name in enumerate(names):
        support = int(confusion[c].sum())
        tp = int(confusion[c, c])
        predicted = int(confusion[:, c].sum())
        recall = tp / support if support else float("nan")
        precision = tp / predicted if predicted else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if support and precision + recall
            else 0.0
        )
        ap = average_precision(scores[:, c], onehot[:, c])
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap": ap,
            "support": float(support),
        }
        if support:
            recalls.append(recall)
            f1s.append(f1)
        if not np.isnan(ap):
            aps.append(ap)

    return EvalResult(
        task="multiclass",
        names=list(names),
        n_samples=int(len(truth)),
        primary=float(np.mean(recalls)) if recalls else 0.0,
        primary_name="balanced_acc",
        scalars={
            "top1": top1,
            f"top{k}": topk,
            "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
            "map": float(np.mean(aps)) if aps else float("nan"),
        },
        per_class=per_class,
        confusion=confusion,
    )


def evaluate(
    logits: np.ndarray,
    targets: np.ndarray,
    names: list[str],
    task: str = "multilabel",
    **kwargs,
) -> EvalResult:
    if task == "multiclass":
        return evaluate_multiclass(logits, targets, names)
    return evaluate_multilabel(logits, targets, names, **kwargs)


def format_confusion(confusion: np.ndarray, names: list[str], max_width: int = 6) -> str:
    """Row-normalised confusion matrix, printed compactly."""
    totals = confusion.sum(axis=1, keepdims=True)
    normalised = np.divide(confusion, np.maximum(totals, 1))
    label_width = max(len(n) for n in names) + 1
    header = " " * label_width + "".join(n[:max_width].rjust(max_width + 1) for n in names)
    lines = [header]
    for i, name in enumerate(names):
        cells = "".join(
            (f"{normalised[i, j]:{max_width + 1}.2f}" if normalised[i, j] else " " * (max_width + 1))
            for j in range(len(names))
        )
        lines.append(name.ljust(label_width) + cells)
    return "\n".join(lines)
