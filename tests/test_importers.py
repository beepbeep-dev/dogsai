from __future__ import annotations

import json

import pytest

from dogsai.importers import (
    from_ava_csv,
    from_csv,
    from_cvat,
    from_folders,
    from_label_studio,
    load_label_map,
)


class TestLabelStudio:
    def _export(self, tmp_path, result):
        payload = [{"data": {"video": "clips/yard.mp4"}, "annotations": [{"result": result}]}]
        path = tmp_path / "export.json"
        path.write_text(json.dumps(payload))
        return path

    def test_second_indexed_regions(self, tmp_path):
        path = self._export(
            tmp_path,
            [{"value": {"start": 2.0, "end": 5.5, "labels": ["running", "barking"]}}],
        )
        result = from_label_studio(path)
        assert len(result) == 1
        annotation = result.annotations[0]
        assert annotation.labels == ["running", "barking"]
        assert (annotation.start, annotation.end) == (2.0, 5.5)
        assert annotation.group == "yard"

    def test_frame_indexed_timeline_labels(self, tmp_path):
        path = self._export(
            tmp_path,
            [{"value": {"timelinelabels": ["digging"], "ranges": [{"start": 30, "end": 60}]}}],
        )
        result = from_label_studio(path, default_fps=30.0)
        annotation = result.annotations[0]
        assert annotation.labels == ["digging"]
        assert annotation.start == pytest.approx(1.0)
        assert annotation.end == pytest.approx(2.0)

    def test_multiple_ranges_become_multiple_spans(self, tmp_path):
        path = self._export(
            tmp_path,
            [{"value": {"timelinelabels": ["barking"],
                        "ranges": [{"start": 0, "end": 30}, {"start": 60, "end": 90}]}}],
        )
        assert len(from_label_studio(path, default_fps=30.0)) == 2

    def test_cancelled_annotations_are_skipped(self, tmp_path):
        payload = [{
            "data": {"video": "a.mp4"},
            "annotations": [{"was_cancelled": True,
                             "result": [{"value": {"start": 0, "end": 1, "labels": ["running"]}}]}],
        }]
        path = tmp_path / "e.json"
        path.write_text(json.dumps(payload))
        assert len(from_label_studio(path)) == 0

    def test_label_map_renames_and_drops(self, tmp_path):
        path = self._export(
            tmp_path,
            [{"value": {"start": 0, "end": 1, "labels": ["Run", "Unknown Thing"]}}],
        )
        result = from_label_studio(path, label_map={"run": "running"})
        assert result.annotations[0].labels == ["running"]
        assert result.unmapped["Unknown Thing"] == 1
        assert "DROPPED" in result.report()

    def test_keep_unmapped_normalises_names(self, tmp_path):
        path = self._export(tmp_path, [{"value": {"start": 0, "end": 1, "labels": ["Tail Wagging"]}}])
        result = from_label_studio(path, label_map={"run": "running"}, keep_unmapped=True)
        assert result.annotations[0].labels == ["tail_wagging"]

    def test_resolves_hash_prefixed_upload_paths(self, tmp_path):
        (tmp_path / "videos").mkdir()
        (tmp_path / "videos" / "yard.mp4").write_bytes(b"x")
        payload = [{
            "data": {"video": "/data/upload/1/a1b2c3d4-yard.mp4"},
            "annotations": [{"result": [{"value": {"start": 0, "end": 1, "labels": ["running"]}}]}],
        }]
        path = tmp_path / "e.json"
        path.write_text(json.dumps(payload))
        result = from_label_studio(path, video_root=tmp_path / "videos")
        assert result.annotations[0].video.endswith("videos/yard.mp4")


class TestCsv:
    def test_basic_columns(self, tmp_path):
        path = tmp_path / "a.csv"
        path.write_text("video,start,end,labels\nclip.mp4,1.0,3.0,running|barking\n")
        result = from_csv(path)
        assert result.annotations[0].labels == ["running", "barking"]
        assert result.annotations[0].start == 1.0

    def test_column_aliases(self, tmp_path):
        path = tmp_path / "a.csv"
        path.write_text("path,begin,stop,behaviour,dog\nclip.mp4,0,2,sitting,rex\n")
        annotation = from_csv(path).annotations[0]
        assert annotation.labels == ["sitting"]
        assert annotation.subject == "rex"

    def test_missing_end_means_whole_clip(self, tmp_path):
        path = tmp_path / "a.csv"
        path.write_text("video,labels\nclip.mp4,running\n")
        annotation = from_csv(path).annotations[0]
        assert annotation.end is None

    def test_explicit_group_is_kept(self, tmp_path):
        path = tmp_path / "a.csv"
        path.write_text("video,labels,group\na.mp4,running,session_9\n")
        assert from_csv(path).annotations[0].group_key == "session_9"

    def test_requires_video_and_label_columns(self, tmp_path):
        path = tmp_path / "a.csv"
        path.write_text("a,b\n1,2\n")
        with pytest.raises(ValueError, match="labels column"):
            from_csv(path)


class TestAva:
    def test_rows_at_the_same_timestamp_merge_into_one_span(self, tmp_path):
        path = tmp_path / "a.csv"
        path.write_text(
            "vid1,10.0,0.1,0.1,0.9,0.9,running\n"
            "vid1,10.0,0.1,0.1,0.9,0.9,barking\n"
            "vid1,20.0,0.1,0.1,0.9,0.9,sitting\n"
        )
        result = from_ava_csv(path, keep_unmapped=True, window=2.0)
        assert len(result) == 2
        first = result.annotations[0]
        assert sorted(first.labels) == ["barking", "running"]
        assert first.start == pytest.approx(9.0)
        assert first.end == pytest.approx(11.0)

    def test_numeric_action_ids_are_mapped(self, tmp_path):
        path = tmp_path / "a.csv"
        path.write_text("vid1,5.0,0,0,1,1,3\n")
        result = from_ava_csv(path, action_names={3: "digging"}, keep_unmapped=True)
        assert result.annotations[0].labels == ["digging"]

    def test_negative_timestamps_are_clamped(self, tmp_path):
        path = tmp_path / "a.csv"
        path.write_text("vid1,0.2,0,0,1,1,running\n")
        assert from_ava_csv(path, keep_unmapped=True, window=2.0).annotations[0].start == 0.0


class TestCvat:
    def test_consecutive_tag_frames_collapse_into_one_span(self, tmp_path):
        path = tmp_path / "a.xml"
        path.write_text(
            "<annotations><meta><task><name>walk.mp4</name>"
            "<original_size></original_size><fps>25</fps></task></meta>"
            + "".join(f'<tag frame="{i}" label="walking"/>' for i in range(10))
            + "</annotations>"
        )
        result = from_cvat(path, keep_unmapped=True)
        assert len(result) == 1
        assert result.annotations[0].labels == ["walking"]
        assert result.annotations[0].start == pytest.approx(0.0)
        assert result.annotations[0].end == pytest.approx(10 / 25)

    def test_gaps_split_into_separate_spans(self, tmp_path):
        frames = list(range(5)) + list(range(20, 25))
        path = tmp_path / "a.xml"
        path.write_text(
            "<annotations><meta><task><name>a.mp4</name><fps>25</fps></task></meta>"
            + "".join(f'<tag frame="{i}" label="barking"/>' for i in frames)
            + "</annotations>"
        )
        assert len(from_cvat(path, keep_unmapped=True)) == 2


class TestFolders:
    def test_directory_names_become_labels(self, mini_dataset, tmp_path):
        source = next((mini_dataset / "videos").glob("*.mp4"))
        for name in ("running", "digging"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "clip.mp4").write_bytes(source.read_bytes())
        result = from_folders(tmp_path)
        assert {a.labels[0] for a in result.annotations} == {"running", "digging"}


def test_load_label_map_normalises_keys(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"Tail Wag": "tail_wagging", "run-fast": "running"}))
    mapping = load_label_map(path)
    assert mapping == {"tail_wag": "tail_wagging", "run_fast": "running"}


def test_load_label_map_of_none_is_none():
    assert load_label_map(None) is None
