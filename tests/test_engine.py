from __future__ import annotations

import numpy as np
import pytest
import torch

from dogsai.config import Config
from dogsai.engine import (
    FocalBCELoss,
    SoftTargetCrossEntropy,
    cosine_lr,
    load_checkpoint,
    param_groups,
    save_checkpoint,
)
from dogsai.labels import LabelSpace
from dogsai.model import build_model


class TestSoftTargetCrossEntropy:
    def test_confident_correct_prediction_has_low_loss(self):
        criterion = SoftTargetCrossEntropy()
        logits = torch.tensor([[10.0, -10.0]])
        targets = torch.tensor([[1.0, 0.0]])
        assert criterion(logits, targets).item() < 1e-3

    def test_confident_wrong_prediction_has_high_loss(self):
        criterion = SoftTargetCrossEntropy()
        logits = torch.tensor([[10.0, -10.0]])
        targets = torch.tensor([[0.0, 1.0]])
        assert criterion(logits, targets).item() > 10.0

    def test_smoothing_keeps_a_perfect_prediction_from_reaching_zero(self):
        logits = torch.tensor([[20.0, -20.0]])
        targets = torch.tensor([[1.0, 0.0]])
        plain = SoftTargetCrossEntropy(0.0)(logits, targets)
        smoothed = SoftTargetCrossEntropy(0.1)(logits, targets)
        assert smoothed > plain

    def test_accepts_mixup_soft_targets(self):
        criterion = SoftTargetCrossEntropy(0.1)
        loss = criterion(torch.randn(4, 5), torch.full((4, 5), 0.2))
        assert torch.isfinite(loss)

    def test_sample_weights_scale_the_loss(self):
        criterion = SoftTargetCrossEntropy()
        logits = torch.tensor([[3.0, -3.0], [3.0, -3.0]])
        targets = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        unweighted = criterion(logits, targets)
        weighted = criterion(logits, targets, weight=torch.tensor([2.0, 2.0]))
        assert weighted == pytest.approx(unweighted.item() * 2, rel=1e-5)


class TestFocalBCELoss:
    def test_confident_correct_prediction_has_low_loss(self):
        criterion = FocalBCELoss(gamma=1.5)
        logits = torch.tensor([[8.0, -8.0]])
        targets = torch.tensor([[1.0, 0.0]])
        assert criterion(logits, targets).item() < 1e-3

    def test_focal_term_down_weights_easy_negatives(self):
        """The reason focal loss is here: easy negatives must stop dominating."""
        logits = torch.tensor([[-6.0]])  # confident, correct negative
        targets = torch.tensor([[0.0]])
        plain = FocalBCELoss(gamma=0.0)(logits, targets)
        focal = FocalBCELoss(gamma=2.0)(logits, targets)
        assert focal < plain * 0.1

    def test_hard_examples_are_relatively_preserved(self):
        logits = torch.tensor([[0.0]])  # maximally uncertain
        targets = torch.tensor([[1.0]])
        plain = FocalBCELoss(gamma=0.0)(logits, targets)
        focal = FocalBCELoss(gamma=2.0)(logits, targets)
        assert focal > plain * 0.2  # damped far less than the easy negative above

    def test_pos_weight_increases_the_cost_of_missed_positives(self):
        logits = torch.tensor([[-3.0]])
        targets = torch.tensor([[1.0]])
        plain = FocalBCELoss(gamma=0.0)(logits, targets)
        weighted = FocalBCELoss(gamma=0.0, pos_weight=torch.tensor([5.0]))(logits, targets)
        assert weighted > plain

    def test_gradients_flow(self):
        logits = torch.randn(4, 6, requires_grad=True)
        targets = (torch.rand(4, 6) > 0.7).float()
        FocalBCELoss(gamma=1.5)(logits, targets).backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()

    def test_finite_at_extreme_logits(self):
        criterion = FocalBCELoss(gamma=2.0)
        logits = torch.tensor([[-60.0, 60.0]])
        targets = torch.tensor([[1.0, 0.0]])
        assert torch.isfinite(criterion(logits, targets))


class TestCosineSchedule:
    def test_warms_up_from_near_zero(self):
        assert cosine_lr(0, 100, 10, 1e-3, 0.0) == pytest.approx(1e-4)

    def test_peaks_at_the_end_of_warmup(self):
        assert cosine_lr(9, 100, 10, 1e-3, 0.0) == pytest.approx(1e-3)

    def test_decays_to_min_lr(self):
        assert cosine_lr(100, 100, 10, 1e-3, 1e-6) == pytest.approx(1e-6, abs=1e-9)

    def test_is_monotonic_after_warmup(self):
        values = [cosine_lr(s, 100, 10, 1e-3, 1e-6) for s in range(10, 101)]
        assert all(a >= b - 1e-12 for a, b in zip(values, values[1:]))

    def test_handles_zero_total_steps(self):
        assert cosine_lr(0, 0, 0, 1e-3, 0.0) == 1e-3

    def test_clamps_past_the_end(self):
        assert cosine_lr(500, 100, 10, 1e-3, 1e-6) == pytest.approx(1e-6, abs=1e-9)


class TestParamGroups:
    def test_norms_and_biases_skip_weight_decay(self):
        model = build_model(Config().apply_preset())
        groups = param_groups(model, 0.05)
        assert groups[0]["weight_decay"] == 0.05
        assert groups[1]["weight_decay"] == 0.0
        total = sum(p.numel() for g in groups for p in g["params"])
        assert total == model.num_parameters()

    def test_every_trainable_parameter_lands_in_exactly_one_group(self):
        model = build_model(Config().apply_preset())
        groups = param_groups(model, 0.01)
        seen = [id(p) for g in groups for p in g["params"]]
        assert len(seen) == len(set(seen))
        assert len(seen) == len([p for p in model.parameters() if p.requires_grad])

    def test_attention_query_is_not_decayed(self):
        model = build_model(Config().apply_preset())
        groups = param_groups(model, 0.05)
        query = model.temporal_pool.query
        assert any(p is query for p in groups[1]["params"])


class TestCheckpoints:
    def test_round_trip_preserves_predictions(self, tmp_path):
        config = Config().apply_preset()
        config.model.preset = "custom"
        config.model.width = 0.5
        config.data.clip_frames = 8
        config.data.image_size = 64
        labels = LabelSpace.from_names(["running", "sitting", "barking"])
        model = build_model(config, num_classes=len(labels)).eval()

        path = save_checkpoint(
            tmp_path / "ckpt.pt", model, config, labels,
            epoch=3, thresholds=np.array([0.4, 0.5, 0.6]),
        )
        restored, restored_config, restored_labels, extra = load_checkpoint(path)

        x = torch.randn(1, 3, 8, 64, 64)
        with torch.no_grad():
            assert torch.allclose(model(x), restored(x), atol=1e-6)
        assert restored_labels.to_list() == labels.to_list()
        assert restored_config.data.clip_frames == 8
        assert extra["epoch"] == 3
        assert extra["thresholds"] == [0.4, 0.5, 0.6]

    def test_checkpoint_carries_its_own_label_space(self, tmp_path):
        config = Config()
        config.model.width = 0.5
        labels = LabelSpace.from_names(["a", "b"])
        model = build_model(config, num_classes=2)
        path = save_checkpoint(tmp_path / "c.pt", model, config, labels)
        _, _, restored, _ = load_checkpoint(path)
        assert restored.to_list() == ["a", "b"]

    def test_ema_weights_are_preferred_when_present(self, tmp_path):
        from dogsai.model import ModelEMA

        config = Config()
        config.model.width = 0.5
        labels = LabelSpace.from_names(["a", "b"])
        model = build_model(config, num_classes=2).eval()
        ema = ModelEMA(model, decay=0.0, warmup=1)
        with torch.no_grad():
            for param in ema.module.parameters():
                param.fill_(0.01)
        path = save_checkpoint(tmp_path / "c.pt", model, config, labels, ema=ema)

        with_ema, _, _, extra = load_checkpoint(path, prefer_ema=True)
        assert extra["used_ema"] is True
        without, _, _, extra2 = load_checkpoint(path, prefer_ema=False)
        assert extra2["used_ema"] is False
        x = torch.randn(1, 3, 8, 64, 64)
        with torch.no_grad():
            assert not torch.allclose(with_ema(x), without(x))


class TestLabelSpace:
    def test_index_and_name_round_trip(self):
        labels = LabelSpace.from_names(["a", "b", "c"])
        assert labels.index("b") == 1
        assert labels.name(1) == "b"

    def test_unknown_name_raises_keyerror(self):
        with pytest.raises(KeyError):
            LabelSpace.from_names(["a"]).index("z")

    def test_duplicates_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            LabelSpace.from_names(["a", "a"])

    def test_empty_is_rejected(self):
        with pytest.raises(ValueError):
            LabelSpace.from_names([])

    def test_posture_and_action_indices_partition_the_default_space(self):
        labels = LabelSpace.default()
        assert set(labels.posture_indices).isdisjoint(labels.action_indices)
        assert len(labels.posture_indices) + len(labels.action_indices) == len(labels)


class TestConfig:
    def test_presets_set_resolution_and_width(self):
        for preset, size in (("nano", 128), ("small", 160), ("base", 192)):
            config = Config()
            config.model.preset = preset
            config.apply_preset()
            assert config.data.image_size == size

    def test_round_trips_through_dict(self):
        config = Config().apply_preset()
        config.train.epochs = 7
        restored = Config.from_dict(config.to_dict())
        assert restored.train.epochs == 7
        assert restored.data.image_size == config.data.image_size

    def test_save_and_load(self, tmp_path):
        config = Config().apply_preset()
        config.train.lr = 1e-5
        config.save(tmp_path / "c.json")
        assert Config.load(tmp_path / "c.json").train.lr == 1e-5

    def test_unknown_key_is_rejected(self):
        with pytest.raises(KeyError):
            Config.from_dict({"not_a_key": 1})

    def test_clip_span_frames(self):
        config = Config()
        config.data.clip_frames = 16
        config.data.frame_stride = 2
        assert config.clip_span_frames == 32
