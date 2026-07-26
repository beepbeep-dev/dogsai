from __future__ import annotations

import numpy as np
import torch

from dogsai.transforms import (
    MEAN,
    STD,
    ClipTransform,
    center_crop,
    colour_jitter,
    denormalise,
    hflip,
    mixup_batch,
    random_erase,
    random_resized_crop,
    to_greyscale,
    to_tensor,
)


def make_clip(t=8, h=60, w=80) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(t, h, w, 3), dtype=np.uint8)


def test_to_tensor_layout_and_normalisation():
    clip = make_clip()
    tensor = to_tensor(clip)
    assert tensor.shape == (3, 8, 60, 80)  # C, T, H, W
    assert tensor.dtype == torch.float32
    # A mid-grey pixel must map to (0.5 - mean) / std per channel.
    grey = np.full((1, 2, 2, 3), 128, dtype=np.uint8)
    got = to_tensor(grey)[:, 0, 0, 0].numpy()
    expected = [(128 / 255 - m) / s for m, s in zip(MEAN, STD)]
    assert np.allclose(got, expected, atol=1e-5)


def test_denormalise_round_trips():
    clip = make_clip(t=4, h=8, w=8)
    restored = denormalise(to_tensor(clip)) * 255
    assert np.allclose(restored.permute(1, 2, 3, 0).numpy(), clip, atol=1.5)


def test_center_crop_is_square_and_deterministic():
    clip = make_clip(h=60, w=100)
    a = center_crop(clip, 32)
    b = center_crop(clip, 32)
    assert a.shape == (8, 32, 32, 3)
    assert np.array_equal(a, b)


def test_center_crop_upscales_small_input():
    clip = make_clip(h=20, w=24)
    assert center_crop(clip, 64).shape == (8, 64, 64, 3)


def test_random_resized_crop_output_shape():
    clip = make_clip()
    out = random_resized_crop(clip, 32, rng=np.random.default_rng(1))
    assert out.shape == (8, 32, 32, 3)


def test_geometric_augmentation_is_identical_across_frames():
    """The core video-augmentation invariant: one crop for the whole clip.

    Built from a clip where every frame is the same image, so any per-frame
    variation in the transform shows up as variation in the output.
    """
    single = make_clip(t=1, h=60, w=80)
    clip = np.repeat(single, 10, axis=0)
    out = random_resized_crop(clip, 32, rng=np.random.default_rng(5))
    for frame in out[1:]:
        assert np.array_equal(frame, out[0])


def test_colour_jitter_is_identical_across_frames():
    single = make_clip(t=1, h=20, w=20)
    clip = np.repeat(single, 6, axis=0)
    out = colour_jitter(clip, 0.4, rng=np.random.default_rng(2))
    for frame in out[1:]:
        assert np.array_equal(frame, out[0])


def test_colour_jitter_changes_pixels_but_keeps_range():
    clip = make_clip(t=2, h=16, w=16)
    out = colour_jitter(clip, 0.5, rng=np.random.default_rng(3))
    assert out.dtype == np.uint8
    assert out.shape == clip.shape
    assert not np.array_equal(out, clip)


def test_colour_jitter_zero_strength_is_identity():
    clip = make_clip()
    assert np.array_equal(colour_jitter(clip, 0.0), clip)


def test_hflip_reverses_width_only():
    clip = make_clip(t=2, h=4, w=5)
    flipped = hflip(clip)
    assert np.array_equal(flipped[:, :, ::-1], clip)


def test_greyscale_has_equal_channels():
    out = to_greyscale(make_clip(t=2, h=8, w=8))
    assert np.array_equal(out[..., 0], out[..., 1])
    assert np.array_equal(out[..., 1], out[..., 2])


def test_random_erase_erases_the_same_box_in_every_frame():
    tensor = torch.zeros(3, 6, 40, 40)
    out = random_erase(tensor.clone(), probability=1.0, rng=np.random.default_rng(4))
    touched = (out != 0).any(dim=0)  # (T, H, W)
    for t in range(1, touched.shape[0]):
        assert torch.equal(touched[t], touched[0])
    assert touched[0].any()


def test_random_erase_zero_probability_is_identity():
    tensor = torch.randn(3, 4, 16, 16)
    assert torch.equal(random_erase(tensor.clone(), 0.0), tensor)


def test_eval_transform_is_deterministic_and_ignores_augmentation():
    clip = make_clip()
    transform = ClipTransform(size=32, training=False, hflip_p=1.0, erase_p=1.0)
    a, b = transform(clip), transform(clip)
    assert torch.equal(a, b)


def test_train_transform_varies_between_calls():
    clip = make_clip()
    transform = ClipTransform(size=32, training=True)
    outputs = [transform(clip, rng=np.random.default_rng(i)) for i in range(4)]
    assert not all(torch.equal(outputs[0], o) for o in outputs[1:])
    assert all(o.shape == (3, 8, 32, 32) for o in outputs)


class TestMixup:
    def test_shapes_preserved_and_targets_mixed(self):
        clips = torch.randn(4, 3, 6, 16, 16)
        targets = torch.eye(4)
        mixed, mixed_targets = mixup_batch(clips, targets, alpha=1.0)
        assert mixed.shape == clips.shape
        assert mixed_targets.shape == targets.shape
        # Convex combination: rows still sum to 1 for one-hot inputs.
        assert torch.allclose(mixed_targets.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_alpha_zero_is_identity(self):
        clips = torch.randn(2, 3, 4, 8, 8)
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        mixed, mixed_targets = mixup_batch(clips.clone(), targets.clone(), alpha=0.0)
        assert torch.equal(mixed, clips)
        assert torch.equal(mixed_targets, targets)

    def test_single_sample_batch_is_untouched(self):
        clips = torch.randn(1, 3, 4, 8, 8)
        targets = torch.ones(1, 3)
        mixed, _ = mixup_batch(clips.clone(), targets, alpha=1.0)
        assert torch.equal(mixed, clips)
