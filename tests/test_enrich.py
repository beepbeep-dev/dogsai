from __future__ import annotations

import json

import pytest

from dogsai.dataset import Annotation, load_annotations, save_annotations
from dogsai.enrich import (
    ADDED_LABEL,
    BARK_KINDS,
    enrich_annotations,
    enrich_dataset,
    load_audio_index,
)


def audio(kinds: dict[str, int] | None = None, confidence: float = 0.6) -> dict:
    return {"kinds": kinds or {}, "arousal": 0.5, "valence": 0.0,
            "vocal_fraction": 0.1, "confidence": confidence}


class TestEnrichAnnotations:
    def test_adds_barking_when_bark_events_present(self):
        annotations = [Annotation("a.mp4", ["eating_drinking"])]
        index = {"a.mp4": audio({"bark": 2})}
        enriched, stats = enrich_annotations(annotations, index, compute_missing=False)
        assert enriched[0].labels == ["eating_drinking", "barking"]
        assert stats.added == 1

    def test_leaves_clip_unchanged_when_no_bark_events(self):
        annotations = [Annotation("a.mp4", ["yawning"])]
        index = {"a.mp4": audio({"whine": 3})}  # not a bark kind
        enriched, stats = enrich_annotations(annotations, index, compute_missing=False)
        assert enriched[0].labels == ["yawning"]
        assert stats.no_audio_signal == 1

    def test_does_not_duplicate_an_existing_barking_label(self):
        annotations = [Annotation("a.mp4", ["playing", "barking"])]
        index = {"a.mp4": audio({"bark": 5})}
        enriched, stats = enrich_annotations(annotations, index, compute_missing=False)
        assert enriched[0].labels == ["playing", "barking"]
        assert stats.already_labelled == 1

    def test_missing_audio_entry_is_left_unenriched_when_not_computing(self):
        annotations = [Annotation("a.mp4", ["chewing"])]
        enriched, stats = enrich_annotations(annotations, {}, compute_missing=False)
        assert enriched[0].labels == ["chewing"]
        assert stats.skipped_missing_audio == 1

    def test_min_count_threshold_is_respected(self):
        annotations = [Annotation("a.mp4", ["chewing"])]
        index = {"a.mp4": audio({"bark": 1})}
        enriched, _ = enrich_annotations(
            annotations, index, min_count=3, compute_missing=False
        )
        assert enriched[0].labels == ["chewing"]

    def test_min_confidence_threshold_is_respected(self):
        annotations = [Annotation("a.mp4", ["chewing"])]
        index = {"a.mp4": audio({"bark": 3}, confidence=0.1)}
        enriched, _ = enrich_annotations(
            annotations, index, min_confidence=0.5, compute_missing=False
        )
        assert enriched[0].labels == ["chewing"]

    def test_counts_all_bark_kinds_together(self):
        annotations = [Annotation("a.mp4", ["chewing"])]
        index = {"a.mp4": audio({"bark_alarm": 1, "bark_excited": 1})}
        enriched, _ = enrich_annotations(annotations, index, min_count=2, compute_missing=False)
        assert ADDED_LABEL in enriched[0].labels

    def test_only_bark_kinds_count_not_other_vocalisations(self):
        assert set(BARK_KINDS) == {"bark", "bark_alarm", "bark_excited"}
        annotations = [Annotation("a.mp4", ["chewing"])]
        index = {"a.mp4": audio({"growl": 10, "whine": 10, "howl": 10, "yelp": 10})}
        enriched, stats = enrich_annotations(annotations, index, compute_missing=False)
        assert ADDED_LABEL not in enriched[0].labels
        assert stats.no_audio_signal == 1

    def test_preserves_span_timing_group_and_weight(self):
        annotations = [Annotation("a.mp4", ["chewing"], start=1.0, end=4.0,
                                  group="g1", weight=2.0)]
        index = {"a.mp4": audio({"bark": 2})}
        enriched, _ = enrich_annotations(annotations, index, compute_missing=False)
        result = enriched[0]
        assert (result.start, result.end, result.group, result.weight) == (1.0, 4.0, "g1", 2.0)

    def test_only_ever_adds_never_removes_the_original_label(self):
        """The strict-enrichment guarantee the module promises."""
        annotations = [Annotation("a.mp4", ["playing"])]
        index = {"a.mp4": audio({"bark": 5})}
        enriched, _ = enrich_annotations(annotations, index, compute_missing=False)
        assert "playing" in enriched[0].labels

    def test_meta_records_that_this_clip_was_enriched(self):
        annotations = [Annotation("a.mp4", ["chewing"], meta={"source": "x"})]
        index = {"a.mp4": audio({"bark": 2})}
        enriched, _ = enrich_annotations(annotations, index, compute_missing=False)
        assert enriched[0].meta["audio_enriched"] is True
        assert enriched[0].meta["bark_events"] == 2
        assert enriched[0].meta["source"] == "x"  # original meta preserved

    def test_stats_add_up_to_the_total(self):
        annotations = [
            Annotation("a.mp4", ["chewing"]),
            Annotation("b.mp4", ["playing", "barking"]),
            Annotation("c.mp4", ["yawning"]),
        ]
        index = {"a.mp4": audio({"bark": 3}), "b.mp4": audio({"bark": 3})}
        enriched, stats = enrich_annotations(annotations, index, compute_missing=False)
        assert len(enriched) == 3
        assert stats.added + stats.already_labelled + stats.no_audio_signal + \
            stats.skipped_missing_audio == stats.total == 3

    def test_stats_render_is_human_readable(self):
        annotations = [Annotation("a.mp4", ["chewing"])]
        _, stats = enrich_annotations(annotations, {"a.mp4": audio({"bark": 2})},
                                      compute_missing=False)
        assert "barking" in stats.render()


class TestLoadAudioIndex:
    def test_builds_a_lookup_from_caption_jsonl(self, tmp_path):
        path = tmp_path / "train_captions.jsonl"
        path.write_text(json.dumps({
            "video": "a.mp4", "caption": "x", "labels": ["chewing"],
            "audio": {"kinds": {"bark": 1}},
        }) + "\n")
        index = load_audio_index([path])
        assert index["a.mp4"]["kinds"] == {"bark": 1}

    def test_missing_file_is_skipped_not_an_error(self, tmp_path):
        assert load_audio_index([tmp_path / "nope.jsonl"]) == {}

    def test_multiple_files_merge(self, tmp_path):
        train = tmp_path / "train_captions.jsonl"
        val = tmp_path / "val_captions.jsonl"
        train.write_text(json.dumps({"video": "a.mp4", "audio": {"kinds": {}}}) + "\n")
        val.write_text(json.dumps({"video": "b.mp4", "audio": {"kinds": {}}}) + "\n")
        index = load_audio_index([train, val])
        assert set(index) == {"a.mp4", "b.mp4"}


class TestEnrichDataset:
    def test_writes_enriched_splits_and_a_behaviour_list(self, tmp_path):
        data_root = tmp_path / "data"
        data_root.mkdir()
        save_annotations(data_root / "train.jsonl", [Annotation("a.mp4", ["chewing"])])
        save_annotations(data_root / "val.jsonl", [Annotation("b.mp4", ["playing"])])
        # load_annotations() resolves a relative video path against the file's
        # directory, so the audio index must be keyed the same way real captions
        # are: dogsai.caption_gen writes them from already-resolved Annotations.
        video_a = str(data_root / "a.mp4")
        video_b = str(data_root / "b.mp4")
        captions_dir = data_root / "captions"
        captions_dir.mkdir()
        (captions_dir / "train_captions.jsonl").write_text(
            json.dumps({"video": video_a, "audio": {"kinds": {"bark": 2}, "confidence": 0.6}}) + "\n"
        )
        (captions_dir / "val_captions.jsonl").write_text(
            json.dumps({"video": video_b, "audio": {"kinds": {}, "confidence": 0.0}}) + "\n"
        )

        out_root = tmp_path / "enriched"
        stats = enrich_dataset(data_root, out_root, captions_dir=captions_dir, verbose=False)

        train_out = load_annotations(out_root / "train.jsonl")
        val_out = load_annotations(out_root / "val.jsonl")
        assert train_out[0].labels == ["chewing", "barking"]
        assert val_out[0].labels == ["playing"]
        assert stats["train"].added == 1
        assert stats["val"].added == 0

        names = (out_root / "behaviours.txt").read_text().split()
        assert "barking" in names
        assert "chewing" in names and "playing" in names

    def test_behaviour_list_falls_back_to_existing_file_when_nothing_enriched(self, tmp_path):
        data_root = tmp_path / "data"
        data_root.mkdir()
        (data_root / "behaviours.txt").write_text("chewing\nplaying\n")
        save_annotations(data_root / "train.jsonl", [Annotation("a.mp4", ["chewing"])])
        out_root = tmp_path / "enriched"
        enrich_dataset(data_root, out_root, splits=["train"], verbose=False)
        names = (out_root / "behaviours.txt").read_text().split()
        assert set(names) >= {"chewing", "playing", "barking"}

    def test_skips_a_split_with_no_source_file(self, tmp_path):
        data_root = tmp_path / "data"
        data_root.mkdir()
        save_annotations(data_root / "train.jsonl", [Annotation("a.mp4", ["chewing"])])
        stats = enrich_dataset(data_root, tmp_path / "out", splits=["train", "val"],
                               verbose=False)
        assert set(stats) == {"train"}
