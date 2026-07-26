from __future__ import annotations

import json

import pytest

from dogsai.datasets_hub import (
    REGISTRY,
    convert_dogbehaviour,
    describe_registry,
    download,
    prepare,
)


class TestRegistry:
    def test_every_entry_records_a_licence_and_a_size(self):
        """These are other people's datasets; the terms have to travel with them."""
        for key, spec in REGISTRY.items():
            assert spec.licence, key
            assert spec.approx_gb > 0, key
            assert spec.description, key
            assert spec.repo_id, key

    def test_summary_mentions_licence_and_source(self):
        text = REGISTRY["dogbehaviour"].summary()
        assert "licence" in text
        assert "fishchen/dog-behavior-dataset" in text

    def test_describe_registry_lists_everything(self):
        text = describe_registry()
        for spec in REGISTRY.values():
            assert spec.name in text

    def test_dogbehaviour_labels_map_into_the_taxonomy(self):
        from dogsai.labels import DEFAULT_BEHAVIOURS

        for source, target in REGISTRY["dogbehaviour"].label_map.items():
            assert target in DEFAULT_BEHAVIOURS, f"{source} -> {target} is not in the taxonomy"

    def test_unknown_key_is_rejected(self):
        with pytest.raises(KeyError):
            download("not_a_dataset", "/tmp/nowhere")


@pytest.fixture
def fake_raw(tmp_path, mini_dataset):
    """A miniature copy of the fishchen layout, using real decodable videos."""
    videos = sorted((mini_dataset / "videos").glob("*.mp4"))
    assert len(videos) >= 4
    root = tmp_path / "raw"
    (root / "data" / "videos").mkdir(parents=True)
    records = []
    sources = ["yawn", "eating", "pooping", "toy", "rope"]
    for i, source in enumerate(sources):
        video = videos[i % len(videos)]
        name = f"dog_{i:04d}.mp4"
        (root / "data" / "videos" / name).write_bytes(video.read_bytes())
        records.append({
            "video": f"data/videos/{name}",
            "fps": 25.0,
            "captions": [{"start": 0.5, "end": 2.0, "text": f"Dog is {source}."}],
            "behaviors": [source],
            "tags": [],
        })
    # A record pointing at a file that does not exist must be skipped, not crash.
    records.append({
        "video": "data/videos/missing.mp4", "fps": 25.0,
        "captions": [], "behaviors": ["yawn"], "tags": [],
    })
    (root / "data" / "metadata.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    return root


class TestConvertDogBehaviour:
    def test_maps_source_labels_onto_the_taxonomy(self, fake_raw):
        annotations = convert_dogbehaviour(fake_raw)
        labels = {a.labels[0] for a in annotations}
        assert labels == {"yawning", "eating_drinking", "eliminating", "playing", "chewing"}

    def test_uses_the_caption_span_as_the_interval(self, fake_raw):
        annotation = convert_dogbehaviour(fake_raw)[0]
        assert annotation.start == pytest.approx(0.5)
        assert annotation.end == pytest.approx(2.0)

    def test_missing_videos_are_skipped(self, fake_raw):
        from pathlib import Path

        annotations = convert_dogbehaviour(fake_raw)
        assert len(annotations) == 5  # the sixth record has no file
        # Compare basenames: the pytest tmp dir itself can contain "missing".
        assert all(Path(a.video).name != "missing.mp4" for a in annotations)
        assert all(Path(a.video).exists() for a in annotations)

    def test_each_clip_is_its_own_group(self, fake_raw):
        annotations = convert_dogbehaviour(fake_raw)
        groups = [a.group_key for a in annotations]
        assert len(set(groups)) == len(groups)

    def test_raw_labels_are_retained_for_traceability(self, fake_raw):
        annotation = convert_dogbehaviour(fake_raw)[0]
        assert annotation.meta["raw_labels"] == ["yawn"]
        assert "fishchen" in annotation.meta["source"]

    def test_padding_widens_the_span(self, fake_raw):
        padded = convert_dogbehaviour(fake_raw, pad=0.4)[0]
        assert padded.start == pytest.approx(0.1)
        assert padded.end == pytest.approx(2.4)

    def test_padding_never_goes_negative(self, fake_raw):
        annotations = convert_dogbehaviour(fake_raw, pad=99.0)
        assert all(a.start >= 0.0 for a in annotations)

    def test_custom_label_map_is_respected(self, fake_raw):
        annotations = convert_dogbehaviour(fake_raw, label_map={"yawn": "yawning"})
        # unmapped names fall through lowercased rather than vanishing
        assert {a.labels[0] for a in annotations} >= {"yawning", "eating", "toy"}

    def test_missing_metadata_is_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="metadata"):
            convert_dogbehaviour(tmp_path)


class TestPrepare:
    def test_writes_splits_and_a_behaviour_list(self, fake_raw, tmp_path):
        out = tmp_path / "prepared"
        written = prepare("dogbehaviour", fake_raw, out,
                          fractions={"train": 0.6, "val": 0.4}, seed=0)
        assert (out / "train.jsonl").exists()
        assert (out / "val.jsonl").exists()
        names = (out / "behaviours.txt").read_text().split()
        assert set(names) == {"chewing", "eating_drinking", "eliminating",
                              "playing", "yawning"}
        assert set(written) >= {"train", "val", "behaviours"}

    def test_splits_are_group_disjoint(self, fake_raw, tmp_path):
        from dogsai.dataset import load_annotations

        out = tmp_path / "prepared"
        prepare("dogbehaviour", fake_raw, out, fractions={"train": 0.6, "val": 0.4}, seed=0)
        train = {a.group_key for a in load_annotations(out / "train.jsonl")}
        val = {a.group_key for a in load_annotations(out / "val.jsonl")}
        assert train.isdisjoint(val)

    def test_every_annotation_survives_the_split(self, fake_raw, tmp_path):
        from dogsai.dataset import load_annotations

        out = tmp_path / "prepared"
        prepare("dogbehaviour", fake_raw, out, fractions={"train": 0.6, "val": 0.4}, seed=0)
        total = len(load_annotations(out / "train.jsonl")) + len(
            load_annotations(out / "val.jsonl")
        )
        assert total == 5

    def test_prepared_splits_load_as_a_dataset(self, fake_raw, tmp_path):
        from dogsai.config import DataConfig
        from dogsai.dataset import ClipDataset, discover_split
        from dogsai.labels import LabelSpace

        out = tmp_path / "prepared"
        prepare("dogbehaviour", fake_raw, out, fractions={"train": 1.0, "val": 0.0}, seed=0)
        labels = LabelSpace.from_names(
            (out / "behaviours.txt").read_text().split()
        )
        config = DataConfig(clip_frames=4, frame_stride=2, image_size=32,
                            num_workers=0, cache_index=False)
        dataset = ClipDataset(discover_split(out, "train"), labels, config, "multiclass")
        assert len(dataset) > 0
        assert dataset[0]["clip"].shape == (3, 4, 32, 32)

    def test_unknown_dataset_key_is_rejected(self, tmp_path):
        with pytest.raises(KeyError):
            prepare("nope", tmp_path, tmp_path / "out")

    def test_empty_conversion_is_an_error(self, tmp_path):
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "metadata.jsonl").write_text("")
        with pytest.raises(ValueError, match="zero annotations"):
            prepare("dogbehaviour", tmp_path, tmp_path / "out")
