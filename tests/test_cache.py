from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from dogsai.cache import (
    CacheSpec,
    CachedClipDataset,
    build_cache,
    build_split_caches,
    cache_is_valid,
    estimate_cache_size,
)
from dogsai.config import DataConfig
from dogsai.dataset import Annotation, discover_split
from dogsai.labels import LabelSpace

SPEC = CacheSpec(frames=8, size=48)


def data_config(**kwargs) -> DataConfig:
    defaults = dict(clip_frames=4, frame_stride=1, image_size=32,
                    num_workers=0, cache_index=False)
    defaults.update(kwargs)
    return DataConfig(**defaults)


@pytest.fixture(scope="module")
def cached(tmp_path_factory, mini_dataset):
    annotations = discover_split(mini_dataset, "train")
    out = tmp_path_factory.mktemp("cache") / "train"
    build_cache(annotations, out, SPEC, workers=2, verbose=False)
    return out, annotations


class TestBuildCache:
    def test_writes_pixels_and_an_index(self, cached):
        out, annotations = cached
        assert (out / "clips.npy").exists()
        assert (out / "index.json").exists()
        payload = json.loads((out / "index.json").read_text())
        assert payload["count"] == len(annotations)
        assert payload["spec"]["frames"] == SPEC.frames

    def test_array_has_the_declared_shape_and_dtype(self, cached):
        out, annotations = cached
        clips = np.load(out / "clips.npy", mmap_mode="r")
        assert clips.shape == (len(annotations), SPEC.frames, SPEC.size, SPEC.size, 3)
        assert clips.dtype == np.uint8

    def test_cached_pixels_are_real_frames_not_zeros(self, cached):
        out, _ = cached
        clips = np.load(out / "clips.npy", mmap_mode="r")
        first = np.asarray(clips[0])
        assert first.std() > 1.0

    def test_frames_within_a_clip_differ(self, cached):
        """Proof the temporal axis holds distinct frames, not one repeated frame."""
        out, _ = cached
        clips = np.load(out / "clips.npy", mmap_mode="r")
        clip = np.asarray(clips[0]).astype(np.int32)
        deltas = [np.abs(clip[i + 1] - clip[i]).mean() for i in range(len(clip) - 1)]
        assert max(deltas) > 0.5

    def test_unreadable_videos_are_recorded_not_fatal(self, tmp_path, mini_dataset):
        annotations = discover_split(mini_dataset, "train")[:2]
        annotations.append(Annotation("/nope/missing.mp4", ["running"]))
        out = tmp_path / "c"
        build_cache(annotations, out, SPEC, workers=1, verbose=False)
        payload = json.loads((out / "index.json").read_text())
        assert len(payload["failures"]) == 1
        assert len(payload["usable"]) == 2

    def test_empty_input_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="nothing to cache"):
            build_cache([], tmp_path / "c", SPEC, verbose=False)

    def test_estimate_matches_the_real_file_size(self, cached):
        out, annotations = cached
        estimated = estimate_cache_size(len(annotations), SPEC)
        actual = (out / "clips.npy").stat().st_size / 1e9
        assert estimated == pytest.approx(actual, rel=0.02)


class TestCacheValidity:
    def test_matching_spec_is_valid(self, cached):
        out, _ = cached
        assert cache_is_valid(out, SPEC)

    def test_different_resolution_invalidates(self, cached):
        out, _ = cached
        assert not cache_is_valid(out, CacheSpec(frames=SPEC.frames, size=SPEC.size * 2))

    def test_different_frame_count_invalidates(self, cached):
        out, _ = cached
        assert not cache_is_valid(out, CacheSpec(frames=SPEC.frames + 8, size=SPEC.size))

    def test_missing_cache_is_invalid(self, tmp_path):
        assert not cache_is_valid(tmp_path / "nothing", SPEC)

    def test_corrupt_index_is_invalid(self, tmp_path):
        (tmp_path / "clips.npy").write_bytes(b"x")
        (tmp_path / "index.json").write_text("{not json")
        assert not cache_is_valid(tmp_path, SPEC)


class TestCachedClipDataset:
    def _dataset(self, cached, **kwargs):
        out, _ = cached
        return CachedClipDataset(
            out, LabelSpace.default(), data_config(), "multilabel", **kwargs
        )

    def test_yields_model_ready_clips(self, cached):
        dataset = self._dataset(cached)
        item = dataset[0]
        assert item["clip"].shape == (3, 4, 32, 32)
        assert item["clip"].dtype == torch.float32
        assert item["target"].shape == (len(LabelSpace.default()),)

    def test_targets_match_the_source_annotations(self, cached):
        out, annotations = cached
        dataset = self._dataset(cached)
        labels = LabelSpace.default()
        expected = {tuple(sorted(a.labels)) for a in annotations}
        got = {
            tuple(sorted(labels.name(i) for i in np.flatnonzero(t)))
            for t in dataset.targets
        }
        assert got <= expected

    def test_training_jitters_the_temporal_window(self, cached):
        """The point of caching more frames than needed."""
        dataset = self._dataset(cached, training=True)
        clips = [dataset[0]["clip"] for _ in range(6)]
        assert not all(torch.equal(clips[0], c) for c in clips[1:])

    def test_eval_access_is_deterministic(self, cached):
        dataset = self._dataset(cached, training=False)
        assert torch.equal(dataset[0]["clip"], dataset[0]["clip"])

    def test_rejects_resolution_larger_than_the_cache(self, cached):
        out, _ = cached
        with pytest.raises(ValueError, match="rebuild the cache"):
            CachedClipDataset(out, LabelSpace.default(),
                              data_config(image_size=SPEC.size * 2), "multilabel")

    def test_rejects_more_frames_than_the_cache_holds(self, cached):
        out, _ = cached
        with pytest.raises(ValueError, match="rebuild"):
            CachedClipDataset(out, LabelSpace.default(),
                              data_config(clip_frames=SPEC.frames + 4), "multilabel")

    def test_exposes_the_interface_samplers_rely_on(self, cached):
        dataset = self._dataset(cached)
        assert len(dataset.positives_per_class()) == len(LabelSpace.default())
        assert len(dataset.groups()) == len(dataset)
        assert len(dataset.samples) == len(dataset)
        assert dataset.label_counts()

    def test_works_with_the_repeat_factor_sampler(self, cached):
        from dogsai.dataset import RepeatFactorSampler

        dataset = self._dataset(cached)
        sampler = RepeatFactorSampler(dataset, seed=0)
        indices = list(iter(sampler))
        assert indices and all(0 <= i < len(dataset) for i in indices)

    def test_multiclass_encoding_rejects_multi_label_records(self, cached):
        out, _ = cached
        # The synthetic sessions carry co-occurring labels, so multiclass must
        # either drop them or refuse; it must never silently pick one.
        try:
            dataset = CachedClipDataset(
                out, LabelSpace.default(), data_config(), "multiclass"
            )
        except ValueError:
            return  # every record was multi-label; refusing is correct
        assert all(t.sum() == 1 for t in dataset.targets)
        assert dataset.skipped


class TestBuildSplitCaches:
    def test_creates_one_cache_per_split(self, tmp_path, mini_dataset):
        splits = {name: discover_split(mini_dataset, name) for name in ("train", "val")}
        dirs = build_split_caches(splits, tmp_path, SPEC, workers=2, verbose=False)
        assert set(dirs) == {"train", "val"}
        for path in dirs.values():
            assert (path / "clips.npy").exists()

    def test_reuses_a_valid_cache_instead_of_rebuilding(self, tmp_path, mini_dataset):
        splits = {"train": discover_split(mini_dataset, "train")}
        first = build_split_caches(splits, tmp_path, SPEC, workers=2, verbose=False)
        stamp = (first["train"] / "clips.npy").stat().st_mtime_ns
        second = build_split_caches(splits, tmp_path, SPEC, workers=2, verbose=False)
        assert (second["train"] / "clips.npy").stat().st_mtime_ns == stamp

    def test_force_rebuilds(self, tmp_path, mini_dataset):
        splits = {"train": discover_split(mini_dataset, "train")}
        first = build_split_caches(splits, tmp_path, SPEC, workers=2, verbose=False)
        stamp = (first["train"] / "clips.npy").stat().st_mtime_ns
        second = build_split_caches(splits, tmp_path, SPEC, workers=2, verbose=False, force=True)
        assert (second["train"] / "clips.npy").stat().st_mtime_ns != stamp

    def test_cached_training_reaches_the_same_shapes_as_uncached(self, tmp_path, mini_dataset):
        """Cached and uncached datasets must be interchangeable to the Trainer."""
        from dogsai.dataset import ClipDataset

        annotations = discover_split(mini_dataset, "train")
        config = data_config()
        labels = LabelSpace.default()
        uncached = ClipDataset(annotations, labels, config, "multilabel")
        dirs = build_split_caches({"train": annotations}, tmp_path, SPEC, workers=2, verbose=False)
        cached_dataset = CachedClipDataset(dirs["train"], labels, config, "multilabel")
        assert cached_dataset[0]["clip"].shape == uncached[0]["clip"].shape
        assert cached_dataset[0]["target"].shape == uncached[0]["target"].shape
