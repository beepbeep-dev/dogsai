from __future__ import annotations

import pytest
import torch

from dogsai.config import Config
from dogsai.model import (
    DogBehaviourNet,
    DropPath,
    FactorisedBlock,
    ModelEMA,
    MotionStem,
    SqueezeExcite3d,
    TemporalAttentionPool,
    build_model,
    make_divisible,
)


class TestBlocks:
    def test_make_divisible_rounds_to_multiples_of_eight(self):
        assert make_divisible(24) == 24
        assert make_divisible(32 * 0.75) == 24
        assert make_divisible(1) == 8  # never collapses to zero channels
        for value in (18, 43.2, 112 * 0.75, 176 * 1.25):
            assert make_divisible(value) % 8 == 0

    def test_make_divisible_never_loses_more_than_ten_percent(self):
        """The 0.9 guard rounds *up* rather than shaving a layer too thin."""
        assert make_divisible(18) == 24  # 16 would be 11% below the request
        for value in range(1, 400):
            assert make_divisible(value) >= 0.9 * value

    def test_motion_stem_doubles_channels_and_preserves_time(self):
        stem = MotionStem()
        x = torch.randn(2, 3, 8, 16, 16)
        out = stem(x)
        assert out.shape == (2, 6, 8, 16, 16)

    def test_motion_stem_is_zero_on_a_static_clip(self):
        """A frozen clip has no motion, so the difference channels must vanish."""
        frame = torch.randn(2, 3, 1, 16, 16)
        static = frame.repeat(1, 1, 8, 1, 1)
        out = MotionStem()(static)
        assert torch.allclose(out[:, 3:], torch.zeros_like(out[:, 3:]), atol=1e-6)

    def test_motion_stem_first_timestep_is_padded_not_dropped(self):
        x = torch.randn(1, 3, 5, 8, 8)
        out = MotionStem()(x)
        assert out.shape[2] == 5
        assert torch.allclose(out[:, 3:, 0], torch.zeros(1, 3, 8, 8), atol=1e-6)

    def test_squeeze_excite_preserves_shape_and_gates(self):
        se = SqueezeExcite3d(16)
        x = torch.randn(2, 16, 4, 8, 8)
        out = se(x)
        assert out.shape == x.shape
        assert not torch.equal(out, x)

    def test_drop_path_is_identity_in_eval(self):
        layer = DropPath(0.9).eval()
        x = torch.randn(4, 8, 2, 4, 4)
        assert torch.equal(layer(x), x)

    def test_drop_path_drops_whole_samples_in_train(self):
        layer = DropPath(0.5).train()
        x = torch.ones(64, 4, 2, 2, 2)
        out = layer(x)
        per_sample = out.flatten(1).abs().sum(dim=1)
        # Each sample is either fully dropped or fully kept (rescaled).
        assert set(torch.unique(per_sample).tolist()) <= {0.0, per_sample.max().item()}
        assert (per_sample == 0).any()

    def test_factorised_block_residual_starts_as_identity(self):
        """Zero-initialised projection BN means a fresh residual block is a no-op."""
        block = FactorisedBlock(16, 16, stride=1).eval()
        x = torch.randn(2, 16, 4, 8, 8)
        assert torch.allclose(block(x), x, atol=1e-5)

    def test_factorised_block_downsamples(self):
        block = FactorisedBlock(8, 24, stride=2).eval()
        out = block(torch.randn(1, 8, 4, 16, 16))
        assert out.shape == (1, 24, 4, 8, 8)

    def test_factorised_block_without_temporal_kernel(self):
        block = FactorisedBlock(8, 8, temporal_kernel=1).eval()
        assert block(torch.randn(1, 8, 4, 8, 8)).shape == (1, 8, 4, 8, 8)

    def test_temporal_attention_weights_are_a_distribution(self):
        pool = TemporalAttentionPool(32, heads=4)
        pooled, weights = pool(torch.randn(3, 7, 32), return_weights=True)
        assert pooled.shape == (3, 32)
        assert weights.shape == (3, 7)
        assert torch.allclose(weights.sum(dim=1), torch.ones(3), atol=1e-5)
        assert (weights >= 0).all()

    def test_temporal_attention_falls_back_to_one_head_when_indivisible(self):
        pool = TemporalAttentionPool(30, heads=4)
        assert pool.heads == 1


class TestDogBehaviourNet:
    def test_forward_shape(self):
        model = DogBehaviourNet(num_classes=7, width=0.5, depth=0.75).eval()
        logits = model(torch.randn(2, 3, 8, 64, 64))
        assert logits.shape == (2, 7)

    def test_keeps_temporal_resolution_for_pooling(self):
        """Time must not be collapsed before the head, or attention is pointless."""
        model = DogBehaviourNet(num_classes=4, width=0.5).eval()
        sequence = model.forward_features(torch.randn(1, 3, 16, 64, 64))
        assert sequence.ndim == 3  # (B, T', C)
        assert sequence.shape[1] == 4  # 16 -> 8 -> 4 across two temporal strides

    def test_attention_is_returned_per_temporal_position(self):
        model = DogBehaviourNet(num_classes=4, width=0.5).eval()
        logits, attention = model(torch.randn(2, 3, 8, 64, 64), return_attention=True)
        assert logits.shape == (2, 4)
        assert attention.shape[0] == 2
        assert torch.allclose(attention.sum(dim=1), torch.ones(2), atol=1e-4)

    @pytest.mark.parametrize("pool", ["attention", "mean", "max"])
    def test_all_pooling_modes_work(self, pool):
        model = DogBehaviourNet(num_classes=5, width=0.5, temporal_pool=pool).eval()
        assert model(torch.randn(1, 3, 8, 64, 64)).shape == (1, 5)

    @pytest.mark.parametrize("frames", [4, 8, 16])
    def test_variable_clip_length(self, frames):
        model = DogBehaviourNet(num_classes=3, width=0.5).eval()
        assert model(torch.randn(1, 3, frames, 64, 64)).shape == (1, 3)

    @pytest.mark.parametrize("size", [64, 96, 128])
    def test_variable_resolution(self, size):
        model = DogBehaviourNet(num_classes=3, width=0.5).eval()
        assert model(torch.randn(1, 3, 8, size, size)).shape == (1, 3)

    def test_without_motion_stem(self):
        model = DogBehaviourNet(num_classes=3, width=0.5, motion_stem=False).eval()
        assert model(torch.randn(1, 3, 8, 64, 64)).shape == (1, 3)

    def test_rejects_non_5d_input(self):
        model = DogBehaviourNet(num_classes=3, width=0.5).eval()
        with pytest.raises(ValueError, match="5-D"):
            model(torch.randn(2, 3, 64, 64))

    def test_eval_is_deterministic(self):
        model = DogBehaviourNet(num_classes=6, width=0.5, dropout=0.5, drop_path=0.5).eval()
        x = torch.randn(2, 3, 8, 64, 64)
        with torch.no_grad():
            assert torch.equal(model(x), model(x))

    def test_gradients_reach_every_parameter(self):
        model = DogBehaviourNet(num_classes=4, width=0.5)
        model(torch.randn(2, 3, 8, 64, 64)).sum().backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert missing == []

    def test_prior_bias_matches_positive_rate(self):
        model = DogBehaviourNet(num_classes=3, width=0.5)
        model.set_prior_bias([0.5, 0.1, 0.01])
        probabilities = torch.sigmoid(model.classifier.bias)
        assert torch.allclose(probabilities, torch.tensor([0.5, 0.1, 0.01]), atol=1e-4)

    def test_prior_bias_rejects_wrong_length(self):
        model = DogBehaviourNet(num_classes=3, width=0.5)
        with pytest.raises(ValueError):
            model.set_prior_bias([0.5, 0.5])

    def test_width_multiplier_shrinks_the_model(self):
        small = DogBehaviourNet(num_classes=10, width=0.5).num_parameters()
        large = DogBehaviourNet(num_classes=10, width=1.0).num_parameters()
        assert small < large / 2

    def test_efficiency_budget(self):
        """The whole point of the architecture: guard against silent bloat."""
        model = DogBehaviourNet(num_classes=18, width=0.75, depth=1.0, head_dim=512).eval()
        macs = model.estimate_flops((3, 16, 160, 160))
        assert model.num_parameters() < 5e6
        assert macs < 3e9

    def test_summary_mentions_parameters(self):
        model = DogBehaviourNet(num_classes=4, width=0.5).eval()
        text = model.summary((3, 8, 64, 64))
        assert "parameters" in text and "MACs" in text

    def test_rejects_zero_classes(self):
        with pytest.raises(ValueError):
            DogBehaviourNet(num_classes=0)


def test_build_model_from_config():
    config = Config().apply_preset()
    model = build_model(config)
    assert model.num_classes == config.num_classes


class TestEMA:
    def test_tracks_the_model_towards_new_weights(self):
        model = DogBehaviourNet(num_classes=3, width=0.5)
        ema = ModelEMA(model, decay=0.5, warmup=1)
        with torch.no_grad():
            for param in model.parameters():
                param.add_(1.0)
        before = next(iter(ema.module.parameters())).clone()
        ema.update(model)
        after = next(iter(ema.module.parameters()))
        assert not torch.equal(before, after)

    def test_shadow_parameters_do_not_require_grad(self):
        ema = ModelEMA(DogBehaviourNet(num_classes=3, width=0.5))
        assert all(not p.requires_grad for p in ema.parameters())

    def test_integer_buffers_are_copied_not_averaged(self):
        model = DogBehaviourNet(num_classes=3, width=0.5)
        model.train()
        model(torch.randn(2, 3, 8, 64, 64))  # advances num_batches_tracked
        ema = ModelEMA(model, decay=0.9, warmup=1)
        model(torch.randn(2, 3, 8, 64, 64))
        ema.update(model)
        for key, value in model.state_dict().items():
            if not value.dtype.is_floating_point:
                assert torch.equal(ema.state_dict()[key], value)
