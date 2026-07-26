from __future__ import annotations

import numpy as np
import pytest

from dogsai.audit import (
    audit_annotations,
    audit_splits,
    clip_signature,
    find_near_duplicates,
    hamming,
)
from dogsai.dataset import Annotation, discover_split
from dogsai.labels import LabelSpace


def codes(report) -> set[str]:
    return {f.code for f in report.findings}


@pytest.fixture
def good_annotations(mini_dataset):
    return discover_split(mini_dataset, "train")


class TestAuditBasics:
    def test_clean_dataset_has_no_errors(self, good_annotations):
        report = audit_annotations(good_annotations, LabelSpace.default(), check_duplicates=False)
        assert report.ok
        assert report.n_annotations == len(good_annotations)
        assert report.total_labelled_seconds > 0

    def test_counts_groups_not_just_files(self, good_annotations):
        report = audit_annotations(good_annotations, check_duplicates=False)
        assert report.n_groups <= report.n_annotations
        assert report.n_groups >= 1

    def test_empty_input_is_an_error(self):
        report = audit_annotations([], LabelSpace.default())
        assert not report.ok
        assert "empty" in codes(report)

    def test_missing_file_is_an_error(self, good_annotations):
        annotations = list(good_annotations) + [Annotation("/nope/gone.mp4", ["running"])]
        report = audit_annotations(annotations, check_duplicates=False)
        assert "missing_file" in codes(report)
        assert not report.ok

    def test_unknown_label_is_an_error_with_a_suggestion(self, good_annotations):
        annotations = list(good_annotations)
        annotations[0] = Annotation(annotations[0].video, ["tail_wag"], 0.0, 1.0)
        report = audit_annotations(annotations, LabelSpace.default(), check_duplicates=False)
        assert "unknown_label" in codes(report)
        message = next(f.message for f in report.findings if f.code == "unknown_label")
        assert "tail_wagging" in message  # the typo suggestion fired

    def test_inverted_span_is_an_error(self, good_annotations):
        video = good_annotations[0].video
        report = audit_annotations(
            [Annotation(video, ["running"], start=3.0, end=1.0)], check_duplicates=False
        )
        assert "bad_span" in codes(report)
        assert not report.ok

    def test_span_beyond_video_start_is_an_error(self, good_annotations):
        video = good_annotations[0].video
        report = audit_annotations(
            [Annotation(video, ["running"], start=9999.0, end=10000.0)], check_duplicates=False
        )
        assert "bad_span" in codes(report)

    def test_span_overrunning_the_end_is_a_warning(self, good_annotations):
        video = good_annotations[0].video
        report = audit_annotations(
            [Annotation(video, ["running"], start=0.0, end=9999.0)], check_duplicates=False
        )
        assert "span_overrun" in codes(report)

    def test_span_shorter_than_a_clip_is_a_warning(self, good_annotations):
        video = good_annotations[0].video
        report = audit_annotations(
            [Annotation(video, ["running"], start=0.0, end=0.2)],
            clip_seconds=1.5,
            check_duplicates=False,
        )
        assert "span_too_short" in codes(report)

    def test_missing_class_is_reported(self, good_annotations):
        labels = LabelSpace.from_names(list(LabelSpace.default()) + ["never_seen"])
        report = audit_annotations(good_annotations, labels, check_duplicates=False)
        assert any(
            f.code == "no_examples" and "never_seen" in f.message for f in report.findings
        )

    def test_contradictory_postures_are_flagged(self, good_annotations):
        video = good_annotations[0].video
        report = audit_annotations(
            [
                Annotation(video, ["sitting"], 0.0, 2.0),
                Annotation(video, ["running"], 1.0, 3.0),  # overlaps, other posture
            ],
            LabelSpace.default(),
            check_duplicates=False,
        )
        assert "contradiction" in codes(report)

    def test_non_overlapping_postures_are_not_flagged(self, good_annotations):
        video = good_annotations[0].video
        report = audit_annotations(
            [
                Annotation(video, ["sitting"], 0.0, 1.0),
                Annotation(video, ["running"], 1.5, 2.5),
            ],
            LabelSpace.default(),
            check_duplicates=False,
        )
        assert "contradiction" not in codes(report)

    def test_explicit_negatives_are_informational(self, good_annotations):
        video = good_annotations[0].video
        report = audit_annotations(
            [Annotation(video, [], 0.0, 2.0)], LabelSpace.default(), check_duplicates=False
        )
        assert "negative" in codes(report)
        assert report.ok

    def test_report_renders_and_serialises(self, good_annotations):
        report = audit_annotations(good_annotations, LabelSpace.default(), check_duplicates=False)
        assert "class balance" in report.render()
        payload = report.to_dict()
        assert payload["n_annotations"] == len(good_annotations)
        assert "label_counts" in payload

    def test_imbalance_ratio(self, good_annotations):
        report = audit_annotations(good_annotations, check_duplicates=False)
        assert report.imbalance_ratio() >= 1.0


class TestSignatures:
    def test_identical_clips_have_identical_signatures(self, session_video):
        video, _ = session_video
        from dogsai.video import probe

        meta = probe(video)
        a = clip_signature(video, meta, start=0.0, end=1.0)
        b = clip_signature(video, meta, start=0.0, end=1.0)
        assert hamming(a, b) == 0

    def test_different_intervals_differ(self, session_video):
        video, annotations = session_video
        from dogsai.video import probe

        meta = probe(video)
        a = clip_signature(video, meta, start=annotations[0].start, end=annotations[0].end)
        b = clip_signature(video, meta, start=annotations[-1].start, end=annotations[-1].end)
        assert hamming(a, b) > 0

    def test_hamming_counts_bits(self):
        a = np.array([0b00000000], dtype=np.uint8)
        b = np.array([0b00001111], dtype=np.uint8)
        assert hamming(a, b) == 4

    def test_find_near_duplicates_pairs_identical_entries(self):
        signature = np.array([1, 2, 3], dtype=np.uint8)
        pairs = find_near_duplicates({"a": signature, "b": signature.copy()}, max_distance=0)
        assert len(pairs) == 1 and pairs[0][2] == 0

    def test_find_near_duplicates_respects_the_distance_limit(self):
        a = np.array([0b00000000], dtype=np.uint8)
        b = np.array([0b00001111], dtype=np.uint8)
        assert find_near_duplicates({"a": a, "b": b}, max_distance=3) == []
        assert len(find_near_duplicates({"a": a, "b": b}, max_distance=4)) == 1

    def test_single_entry_has_no_pairs(self):
        assert find_near_duplicates({"a": np.array([1], dtype=np.uint8)}) == []


class TestCrossSplitLeakage:
    def test_shared_group_is_an_error(self, good_annotations):
        shared = good_annotations[0]
        reports, cross = audit_splits(
            {"train": [shared], "val": [shared]},
            LabelSpace.default(),
            check_duplicates=False,
        )
        assert any(f.code == "split_leakage" for f in cross)

    def test_disjoint_groups_pass(self, mini_dataset):
        train = discover_split(mini_dataset, "train")
        val = discover_split(mini_dataset, "val")
        _, cross = audit_splits(
            {"train": train, "val": val}, LabelSpace.default(), check_duplicates=False
        )
        assert not any(f.code == "split_leakage" for f in cross)

    def test_content_duplicate_across_splits_is_caught_despite_different_names(
        self, mini_dataset, tmp_path
    ):
        """The failure a group-key check alone cannot see: same footage, new filename."""
        train = discover_split(mini_dataset, "train")
        original = train[0]
        copy_path = tmp_path / "renamed_and_regrouped.mp4"
        copy_path.write_bytes(open(original.video, "rb").read())
        sneaky = Annotation(
            str(copy_path), list(original.labels), original.start, original.end,
            group="totally_different_group",
        )
        _, cross = audit_splits(
            {"train": [original], "val": [sneaky]},
            LabelSpace.default(),
            check_duplicates=True,
            duplicate_distance=6,
        )
        assert any(f.code == "cross_split_duplicate" for f in cross)
