from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest
import torch

from dogsai.config import DataConfig
from dogsai.dataset import (
    Annotation,
    ClipDataset,
    ClassBalancedSampler,
    RepeatFactorSampler,
    SlidingWindowDataset,
    collate,
    discover_split,
    load_annotations,
    make_splits,
    save_annotations,
)
from dogsai.labels import LabelSpace


def small_data_config(**kwargs) -> DataConfig:
    defaults = dict(clip_frames=6, frame_stride=2, image_size=48, num_workers=0, cache_index=False)
    defaults.update(kwargs)
    return DataConfig(**defaults)


class TestAnnotation:
    def test_group_key_defaults_to_filename_stem(self):
        assert Annotation("a/b/yard_cam.mp4", ["running"]).group_key == "yard_cam"

    def test_explicit_group_wins(self):
        assert Annotation("a.mp4", ["running"], group="session_1").group_key == "session_1"

    def test_round_trips_through_dict(self):
        original = Annotation("clip.mp4", ["running", "barking"], 1.5, 4.0, group="g", weight=2.0)
        restored = Annotation.from_dict(original.to_dict())
        assert restored.video == "clip.mp4"
        assert restored.labels == ["running", "barking"]
        assert restored.start == 1.5 and restored.end == 4.0
        assert restored.group == "g" and restored.weight == 2.0

    def test_accepts_a_single_string_label(self):
        assert Annotation.from_dict({"video": "a.mp4", "label": "running"}).labels == ["running"]

    def test_missing_video_is_an_error(self):
        with pytest.raises(ValueError, match="video"):
            Annotation.from_dict({"labels": ["running"]})

    def test_jsonl_round_trip(self, tmp_path):
        annotations = [
            Annotation("a.mp4", ["running"], 0.0, 2.0, group="g1"),
            Annotation("b.mp4", [], 1.0, 3.0, group="g2"),  # explicit negative
        ]
        path = save_annotations(tmp_path / "train.jsonl", annotations)
        loaded = load_annotations(path, root=tmp_path)
        assert len(loaded) == 2
        assert loaded[1].labels == []

    def test_relative_paths_resolve_against_the_root(self, tmp_path):
        (tmp_path / "train.jsonl").write_text(
            json.dumps({"video": "videos/a.mp4", "labels": ["running"]}) + "\n"
        )
        loaded = load_annotations(tmp_path / "train.jsonl")
        assert loaded[0].video == str(tmp_path / "videos/a.mp4")


class TestClipDataset:
    def test_yields_correctly_shaped_clips(self, mini_dataset):
        annotations = discover_split(mini_dataset, "train")
        labels = LabelSpace.default()
        config = small_data_config()
        dataset = ClipDataset(annotations, labels, config, "multilabel")
        item = dataset[0]
        assert item["clip"].shape == (3, 6, 48, 48)
        assert item["target"].shape == (len(labels),)
        assert item["target"].sum() >= 1

    def test_multilabel_targets_are_multi_hot(self, mini_dataset):
        labels = LabelSpace.from_names(["running", "barking", "sitting"])
        annotations = [
            a for a in discover_split(mini_dataset, "train")
            if any(n in labels for n in a.labels)
        ]
        for annotation in annotations:
            annotation.labels = [n for n in annotation.labels if n in labels]
        dataset = ClipDataset(annotations, labels, small_data_config(), "multilabel")
        assert dataset.positives_per_class().sum() >= len(dataset)

    def test_multiclass_rejects_multi_label_annotations(self, mini_dataset):
        labels = LabelSpace.default()
        annotations = [Annotation(a.video, ["running", "barking"], a.start, a.end)
                       for a in discover_split(mini_dataset, "train")[:2]]
        with pytest.raises(ValueError, match="no usable samples"):
            ClipDataset(annotations, labels, small_data_config(), "multiclass")

    def test_multiclass_strict_mode_raises_the_underlying_error(self, mini_dataset):
        labels = LabelSpace.default()
        annotations = [Annotation(a.video, ["running", "barking"], a.start, a.end)
                       for a in discover_split(mini_dataset, "train")[:2]]
        with pytest.raises(KeyError, match="exactly one label"):
            ClipDataset(annotations, labels, small_data_config(), "multiclass", strict=True)

    def test_unknown_labels_are_skipped_not_fatal(self, mini_dataset):
        annotations = discover_split(mini_dataset, "train")
        labels = LabelSpace.from_names(["running", "sitting", "standing", "walking",
                                        "trotting", "lying_down"])
        for annotation in annotations:
            annotation.labels = annotation.labels + ["not_a_behaviour"]
        with pytest.raises(ValueError, match="no usable samples"):
            ClipDataset(annotations, labels, small_data_config(), "multilabel")

    def test_missing_files_are_reported_in_skipped(self, mini_dataset):
        annotations = discover_split(mini_dataset, "train")
        annotations.append(Annotation("/nope/missing.mp4", ["running"]))
        dataset = ClipDataset(annotations, LabelSpace.default(), small_data_config(), "multilabel")
        assert any("missing.mp4" in path for path, _ in dataset.skipped)

    def test_samples_stay_inside_the_annotated_span(self, mini_dataset):
        """The accuracy-critical invariant of span mode."""
        annotations = discover_split(mini_dataset, "train")
        target = next(a for a in annotations if a.end is not None and a.start > 0)
        dataset = ClipDataset([target], LabelSpace.default(), small_data_config(), "multilabel")
        sample = dataset.samples[0]
        meta = sample.meta
        assert sample.first_frame >= int(target.start * meta.fps)
        assert sample.last_frame <= int(np.ceil(target.end * meta.fps))

    def test_training_draws_different_windows_each_access(self, mini_dataset):
        annotations = discover_split(mini_dataset, "train")[:1]
        dataset = ClipDataset(annotations, LabelSpace.default(), small_data_config(),
                              "multilabel", training=True)
        clips = [dataset[0]["clip"] for _ in range(4)]
        assert not all(torch.equal(clips[0], c) for c in clips[1:])

    def test_eval_access_is_deterministic(self, mini_dataset):
        annotations = discover_split(mini_dataset, "train")[:1]
        dataset = ClipDataset(annotations, LabelSpace.default(), small_data_config(),
                              "multilabel", training=False)
        assert torch.equal(dataset[0]["clip"], dataset[0]["clip"])

    def test_label_counts_match_annotations(self, mini_dataset):
        annotations = discover_split(mini_dataset, "train")
        dataset = ClipDataset(annotations, LabelSpace.default(), small_data_config(), "multilabel")
        expected: Counter = Counter()
        for sample in dataset.samples:
            for name in sample.annotation.labels:
                expected[name] += 1
        assert dataset.label_counts() == expected

    def test_decode_size_never_upscales(self, mini_dataset):
        annotations = discover_split(mini_dataset, "train")[:1]
        config = small_data_config(image_size=512)
        dataset = ClipDataset(annotations, LabelSpace.default(), config, "multilabel")
        assert dataset._decode_size(dataset.samples[0].meta) is None

    def test_decode_size_shrinks_large_video(self, mini_dataset):
        annotations = discover_split(mini_dataset, "train")[:1]
        config = small_data_config(image_size=32)
        dataset = ClipDataset(annotations, LabelSpace.default(), config, "multilabel")
        size = dataset._decode_size(dataset.samples[0].meta)
        assert size is not None and min(size) >= 32
        assert size[0] % 2 == 0 and size[1] % 2 == 0  # H.264 needs even dimensions


class TestCollate:
    def test_stacks_a_batch(self, mini_dataset):
        annotations = discover_split(mini_dataset, "train")
        dataset = ClipDataset(annotations, LabelSpace.default(), small_data_config(), "multilabel")
        batch = collate([dataset[i] for i in range(3)])
        assert batch["clip"].shape[0] == 3
        assert batch["target"].shape == (3, len(LabelSpace.default()))
        assert batch["weight"].shape == (3,)


class TestSlidingWindow:
    def test_covers_the_whole_video(self, session_video):
        video, _ = session_video
        dataset = SlidingWindowDataset(video, small_data_config(), hop_fraction=0.5)
        assert len(dataset) > 1
        first, last = dataset.window_time(0), dataset.window_time(len(dataset) - 1)
        assert first[0] == pytest.approx(0.0)
        assert last[1] == pytest.approx(dataset.meta.duration, abs=0.2)

    def test_windows_overlap_at_half_hop(self, session_video):
        video, _ = session_video
        dataset = SlidingWindowDataset(video, small_data_config(), hop_fraction=0.5)
        a, b = dataset.window_time(0), dataset.window_time(1)
        assert b[0] < a[1]

    def test_items_are_model_ready(self, session_video):
        video, _ = session_video
        dataset = SlidingWindowDataset(video, small_data_config(), hop_fraction=1.0)
        assert dataset[0]["clip"].shape == (3, 6, 48, 48)


class TestSamplers:
    def _dataset(self, mini_dataset):
        return ClipDataset(
            discover_split(mini_dataset, "train"), LabelSpace.default(),
            small_data_config(), "multilabel",
        )

    def test_repeat_factor_oversamples_rare_classes(self, mini_dataset):
        dataset = self._dataset(mini_dataset)
        sampler = RepeatFactorSampler(dataset, seed=0)
        counts = dataset.positives_per_class()
        rarest = int(np.argmin(np.where(counts > 0, counts, np.inf)))
        commonest = int(np.argmax(counts))
        assert sampler.class_factor[rarest] >= sampler.class_factor[commonest]
        assert len(list(iter(sampler))) >= len(dataset)

    def test_repeat_factor_is_epoch_dependent(self, mini_dataset):
        sampler = RepeatFactorSampler(self._dataset(mini_dataset), seed=0)
        sampler.set_epoch(0)
        first = list(iter(sampler))
        sampler.set_epoch(1)
        assert first != list(iter(sampler))

    def test_repeat_factor_indices_are_in_range(self, mini_dataset):
        dataset = self._dataset(mini_dataset)
        sampler = RepeatFactorSampler(dataset, seed=0)
        assert all(0 <= i < len(dataset) for i in sampler)

    def test_class_balanced_weights_sum_to_one(self, mini_dataset):
        sampler = ClassBalancedSampler(self._dataset(mini_dataset), seed=0)
        assert sampler.weights.sum() == pytest.approx(1.0)
        assert len(list(iter(sampler))) == len(sampler.weights)


class TestMakeSplits:
    def test_no_group_crosses_a_split(self):
        annotations = [
            Annotation(f"v{i // 3}.mp4", ["running" if i % 2 else "sitting"], group=f"g{i // 3}")
            for i in range(60)
        ]
        parts = make_splits(annotations, {"train": 0.7, "val": 0.3}, seed=0)
        groups = {name: {a.group_key for a in anns} for name, anns in parts.items()}
        assert groups["train"].isdisjoint(groups["val"])

    def test_all_annotations_are_placed_exactly_once(self):
        annotations = [Annotation(f"v{i}.mp4", ["running"], group=f"g{i}") for i in range(40)]
        parts = make_splits(annotations, {"train": 0.8, "val": 0.2}, seed=1)
        assert sum(len(v) for v in parts.values()) == len(annotations)

    def test_respects_the_requested_ratio_approximately(self):
        annotations = [Annotation(f"v{i}.mp4", ["running"], group=f"g{i}") for i in range(100)]
        parts = make_splits(annotations, {"train": 0.8, "val": 0.2}, seed=2)
        assert 0.65 <= len(parts["train"]) / 100 <= 0.92

    def test_rare_classes_appear_in_every_split(self):
        annotations = [Annotation(f"c{i}.mp4", ["common"], group=f"c{i}") for i in range(90)]
        annotations += [Annotation(f"r{i}.mp4", ["rare"], group=f"r{i}") for i in range(10)]
        parts = make_splits(annotations, {"train": 0.7, "val": 0.3}, seed=0)
        for annotations_for_split in parts.values():
            assert any("rare" in a.labels for a in annotations_for_split)

    def test_supports_three_way_splits(self):
        annotations = [Annotation(f"v{i}.mp4", ["running"], group=f"g{i}") for i in range(90)]
        parts = make_splits(annotations, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=0)
        assert set(parts) == {"train", "val", "test"}
        assert all(len(v) > 0 for v in parts.values())

    def test_rejects_zero_fractions(self):
        with pytest.raises(ValueError):
            make_splits([Annotation("a.mp4", ["x"])], {"train": 0.0, "val": 0.0})


class TestDiscoverSplit:
    def test_prefers_jsonl_over_folders(self, tmp_path):
        (tmp_path / "train").mkdir()
        (tmp_path / "train" / "running").mkdir()
        save_annotations(tmp_path / "train.jsonl", [Annotation("a.mp4", ["sitting"])])
        found = discover_split(tmp_path, "train")
        assert found[0].labels == ["sitting"]

    def test_folder_mode_uses_directory_names_as_labels(self, mini_dataset, tmp_path):
        source = next((mini_dataset / "videos").glob("*.mp4"))
        for name in ("running", "sitting"):
            (tmp_path / "train" / name).mkdir(parents=True)
            (tmp_path / "train" / name / "clip.mp4").write_bytes(source.read_bytes())
        found = discover_split(tmp_path, "train")
        assert {a.labels[0] for a in found} == {"running", "sitting"}

    def test_missing_split_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            discover_split(tmp_path, "nope")
