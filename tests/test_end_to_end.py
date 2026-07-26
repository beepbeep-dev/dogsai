"""End-to-end: synthetic data -> train -> checkpoint -> predict -> spans -> feeling.

These are the tests that catch integration breaks the unit tests cannot see: a
config that does not survive a checkpoint round trip, a sampler that yields
indices the dataset rejects, a label space that shifts between training and
inference. They train real (tiny) models, so they are slower than the rest of the
suite by design.
"""

from __future__ import annotations

import json

import pytest
import torch

from dogsai.config import Config
from dogsai.dataset import ClipDataset, discover_split
from dogsai.engine import Trainer
from dogsai.labels import LabelSpace
from dogsai.predict import BehaviourPredictor


def tiny_config(tmp_path, task="multilabel", behaviours=None) -> Config:
    config = Config()
    config.task = task
    config.behaviours = behaviours or ["lying_down", "standing", "running", "playing"]
    config.model.preset = "custom"
    config.model.width = 0.5
    config.model.depth = 0.5
    config.model.head_dim = 96
    config.data.clip_frames = 6
    config.data.frame_stride = 2
    config.data.image_size = 48
    config.data.num_workers = 0
    config.data.cache_index = False
    config.train.epochs = 2
    config.train.batch_size = 4
    config.train.warmup_epochs = 0.5
    config.train.lr = 3e-3
    config.train.out_dir = str(tmp_path / "run")
    config.train.log_every = 0
    config.train.ema_decay = 0.9
    config.infer.min_duration = 0.0
    config.infer.smooth = 1
    return config


def build_datasets(mini_dataset, config, labels):
    def keep(annotations):
        out = []
        for annotation in annotations:
            filtered = [n for n in annotation.labels if n in labels]
            if filtered:
                annotation.labels = filtered
                out.append(annotation)
        return out

    train = ClipDataset(
        keep(discover_split(mini_dataset, "train")), labels, config.data,
        config.task, training=True,
    )
    val = ClipDataset(
        keep(discover_split(mini_dataset, "val")), labels, config.data,
        config.task, training=False,
    )
    return train, val


@pytest.fixture(scope="module")
def trained(tmp_path_factory, mini_dataset):
    """Train a tiny multilabel model once and reuse it across these tests."""
    tmp_path = tmp_path_factory.mktemp("e2e")
    config = tiny_config(tmp_path)
    labels = LabelSpace.from_names(config.behaviours)
    train, val = build_datasets(mini_dataset, config, labels)
    trainer = Trainer(config, train, val, labels, device="cpu", verbose=False)
    state = trainer.fit()
    return tmp_path / "run" / "best.pt", state, config, labels


class TestTrainingLoop:
    def test_writes_the_expected_artefacts(self, trained):
        checkpoint, _, config, _ = trained
        run_dir = checkpoint.parent
        assert checkpoint.exists()
        assert (run_dir / "last.pt").exists()
        assert (run_dir / "config.json").exists()
        assert (run_dir / "history.json").exists()

    def test_history_records_every_epoch(self, trained):
        checkpoint, _, config, _ = trained
        history = json.loads((checkpoint.parent / "history.json").read_text())
        assert len(history) == config.train.epochs
        assert all("loss" in record for record in history)

    def test_saved_config_reloads(self, trained):
        checkpoint, _, config, _ = trained
        reloaded = Config.load(checkpoint.parent / "config.json")
        assert reloaded.data.clip_frames == config.data.clip_frames
        assert reloaded.task == config.task

    def test_loss_is_finite(self, trained):
        _, state, _, _ = trained
        assert all(record["loss"] == record["loss"] for record in state.history)  # not NaN
        assert all(record["loss"] < 100 for record in state.history)

    def test_tracks_a_best_epoch(self, trained):
        _, state, _, _ = trained
        assert state.best_epoch >= 0


class TestCheckpointToInference:
    def test_predictor_loads_without_the_training_config(self, trained):
        checkpoint, _, config, labels = trained
        predictor = BehaviourPredictor(checkpoint, device="cpu")
        assert predictor.labels.to_list() == labels.to_list()
        assert predictor.config.data.clip_frames == config.data.clip_frames
        assert predictor.config.task == config.task

    def test_thresholds_come_from_the_checkpoint(self, trained):
        checkpoint, _, _, labels = trained
        predictor = BehaviourPredictor(checkpoint, device="cpu")
        assert len(predictor.thresholds) == len(labels)

    def test_predict_produces_a_well_formed_timeline(self, trained, session_video):
        checkpoint, _, _, labels = trained
        video, _ = session_video
        predictor = BehaviourPredictor(checkpoint, device="cpu")
        prediction = predictor.predict(video)

        assert prediction.scores.shape[1] == len(labels)
        assert len(prediction.window_times) == prediction.scores.shape[0]
        assert prediction.meta.duration > 0
        for span in prediction.spans:
            assert span.behaviour in labels
            assert 0.0 <= span.start < span.end <= prediction.meta.duration + 0.5
            assert 0.0 <= span.score <= 1.0

    def test_windows_tile_the_whole_video_in_order(self, trained, session_video):
        checkpoint, _, _, _ = trained
        video, _ = session_video
        prediction = BehaviourPredictor(checkpoint, device="cpu").predict(video)
        starts = [a for a, _ in prediction.window_times]
        assert starts == sorted(starts)
        assert prediction.window_times[0][0] == pytest.approx(0.0)
        assert prediction.window_times[-1][1] == pytest.approx(
            prediction.meta.duration, abs=0.3
        )

    def test_scores_are_probabilities(self, trained, session_video):
        checkpoint, _, _, _ = trained
        video, _ = session_video
        prediction = BehaviourPredictor(checkpoint, device="cpu").predict(video)
        assert prediction.scores.min() >= 0.0
        assert prediction.scores.max() <= 1.0

    def test_json_output_is_serialisable_and_complete(self, trained, session_video):
        checkpoint, _, _, _ = trained
        video, _ = session_video
        prediction = BehaviourPredictor(checkpoint, device="cpu").predict(video)
        payload = json.loads(json.dumps(prediction.to_dict()))
        for key in ("video", "duration", "spans", "behaviours", "affect",
                    "time_per_behaviour", "task"):
            assert key in payload

    def test_timeline_and_str_render(self, trained, session_video):
        checkpoint, _, _, _ = trained
        video, _ = session_video
        prediction = BehaviourPredictor(checkpoint, device="cpu").predict(video)
        assert isinstance(prediction.timeline(), str)
        assert str(prediction)

    def test_affect_read_is_produced_and_hedged(self, trained, session_video):
        checkpoint, _, _, _ = trained
        video, _ = session_video
        prediction = BehaviourPredictor(checkpoint, device="cpu").predict(video)
        reading = prediction.affect()
        assert 0.0 <= reading.confidence <= 1.0
        # A 4-class model must admit how little it can see.
        assert reading.confidence < 1.0
        assert reading.unseen
        assert "not a measurement of emotion" in reading.to_dict()["disclaimer"]

    def test_explain_returns_per_frame_attention(self, trained, session_video):
        checkpoint, _, config, _ = trained
        video, _ = session_video
        predictor = BehaviourPredictor(checkpoint, device="cpu")
        explanation = predictor.explain(video, at=1.0)
        assert len(explanation["top"]) >= 1
        weights = explanation["frame_attention"]
        assert weights and abs(sum(weights) - 1.0) < 1e-3

    def test_score_at_returns_every_class(self, trained, session_video):
        checkpoint, _, _, labels = trained
        video, _ = session_video
        prediction = BehaviourPredictor(checkpoint, device="cpu").predict(video)
        scores = prediction.score_at(0.5)
        assert set(scores) == set(labels.to_list())


class TestMulticlassPath:
    def test_multiclass_trains_and_predicts_one_behaviour_at_a_time(
        self, tmp_path, mini_dataset, session_video
    ):
        config = tiny_config(tmp_path, task="multiclass",
                             behaviours=["lying_down", "standing", "running"])
        config.train.epochs = 1
        labels = LabelSpace.from_names(config.behaviours)

        def single_label(annotations):
            out = []
            for annotation in annotations:
                filtered = [n for n in annotation.labels if n in labels]
                if len(filtered) == 1:
                    annotation.labels = filtered
                    out.append(annotation)
            return out

        train = ClipDataset(single_label(discover_split(mini_dataset, "train")),
                            labels, config.data, "multiclass", training=True)
        val = ClipDataset(single_label(discover_split(mini_dataset, "val")),
                          labels, config.data, "multiclass", training=False)
        trainer = Trainer(config, train, val, labels, device="cpu", verbose=False)
        trainer.fit()

        result = trainer.evaluate()
        assert result is not None
        assert result.confusion is not None
        assert result.primary_name == "balanced_acc"

        video, _ = session_video
        prediction = BehaviourPredictor(tmp_path / "run" / "best.pt", device="cpu").predict(video)
        # argmax semantics: no two behaviours may be active in the same instant
        for a in prediction.spans:
            for b in prediction.spans:
                if a is b or a.behaviour == b.behaviour:
                    continue
                overlap = min(a.end, b.end) - max(a.start, b.start)
                assert overlap <= 1e-6, f"{a.behaviour} overlaps {b.behaviour}"


class TestExportPath:
    def test_torchscript_round_trips(self, trained, tmp_path):
        from dogsai.export import export_torchscript

        checkpoint, _, config, _ = trained
        out = export_torchscript(checkpoint, tmp_path / "model.ts.pt")
        assert out.exists()

        sidecar = json.loads(out.with_suffix(".meta.json").read_text())
        assert sidecar["behaviours"]
        assert sidecar["input"]["clip_frames"] == config.data.clip_frames
        assert sidecar["activation"] in ("sigmoid", "softmax")

        module = torch.jit.load(str(out))
        shape = (1, 3, config.data.clip_frames, config.data.image_size, config.data.image_size)
        with torch.no_grad():
            assert module(torch.randn(*shape)).shape == (1, len(sidecar["behaviours"]))

    def test_benchmark_reports_plausible_numbers(self, trained):
        from dogsai.export import benchmark

        checkpoint, _, _, _ = trained
        stats = benchmark(checkpoint=checkpoint, iterations=3, device="cpu")
        assert stats["latency_ms"] > 0
        assert stats["clips_per_second"] > 0
        assert stats["parameters_m"] > 0
        assert stats["macs_g"] > 0
