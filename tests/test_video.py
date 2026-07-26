from __future__ import annotations

import numpy as np
import pytest

from dogsai.video import (
    VideoError,
    probe,
    read_frames,
    sample_indices,
    window_starts,
)


def test_probe_reads_metadata(tiny_video):
    meta = probe(tiny_video)
    assert meta.n_frames == 60
    assert meta.fps == pytest.approx(30.0, abs=0.01)
    assert (meta.width, meta.height) == (96, 72)
    assert meta.duration == pytest.approx(2.0, abs=0.05)


def test_probe_missing_file():
    with pytest.raises(VideoError):
        probe("/nonexistent/clip.mp4")


def _frame_id(frame: np.ndarray) -> int:
    """Recover the encoded frame index from the red channel."""
    return int(round(float(np.median(frame[:, :, 0])) / 4))


def test_read_frames_returns_requested_indices(tiny_video):
    wanted = [0, 5, 17, 33, 59]
    clip = read_frames(tiny_video, wanted)
    assert clip.shape == (5, 72, 96, 3)
    assert clip.dtype == np.uint8
    # Lossy encoding shifts values slightly; a tolerance of one step is plenty to
    # prove the right frames came back rather than an off-by-N sequence.
    for index, frame in zip(wanted, clip):
        assert abs(_frame_id(frame) - index) <= 1


def test_read_frames_preserves_request_order_and_duplicates(tiny_video):
    clip = read_frames(tiny_video, [9, 3, 3, 40])
    assert clip.shape[0] == 4
    ids = [_frame_id(f) for f in clip]
    assert abs(ids[0] - 9) <= 1
    assert ids[1] == ids[2]
    assert abs(ids[3] - 40) <= 1


def test_read_frames_clamps_past_the_end(tiny_video):
    clip = read_frames(tiny_video, [58, 59, 200, 500])
    assert clip.shape[0] == 4
    assert _frame_id(clip[-1]) >= 55  # padded with the last real frame


def test_read_frames_resizes_during_decode(tiny_video):
    clip = read_frames(tiny_video, [0, 10], size=(48, 36))
    assert clip.shape == (2, 36, 48, 3)


def test_read_frames_seek_path_matches_sequential(tiny_video):
    """Indices past the seek threshold must not shift the returned frames."""
    early = read_frames(tiny_video, [50])
    late = read_frames(tiny_video, [10, 50])
    assert abs(_frame_id(early[0]) - _frame_id(late[1])) <= 1


def test_cv2_backend_agrees_with_av(tiny_video):
    pytest.importorskip("cv2")
    wanted = [1, 12, 44]
    a = read_frames(tiny_video, wanted, backend="av")
    b = read_frames(tiny_video, wanted, backend="cv2")
    for fa, fb in zip(a, b):
        assert abs(_frame_id(fa) - _frame_id(fb)) <= 1


def test_read_frames_rejects_empty_request(tiny_video):
    with pytest.raises(ValueError):
        read_frames(tiny_video, [])


class TestSampleIndices:
    def test_contiguous_respects_stride(self):
        idx = sample_indices(100, 8, stride=3, mode="contiguous", training=False)
        assert len(idx) == 8
        assert np.all(np.diff(idx) == 3)

    def test_tsn_is_sorted_and_in_range(self):
        idx = sample_indices(50, 16, stride=2, mode="tsn", training=False)
        assert len(idx) == 16
        assert np.all(np.diff(idx) >= 0)
        assert idx.min() >= 0 and idx.max() < 50

    def test_eval_sampling_is_deterministic(self):
        a = sample_indices(80, 12, 2, "tsn", training=False)
        b = sample_indices(80, 12, 2, "tsn", training=False)
        assert np.array_equal(a, b)

    def test_training_sampling_varies(self):
        rng = np.random.default_rng(0)
        draws = {tuple(sample_indices(80, 12, 2, "tsn", training=True, rng=rng)) for _ in range(8)}
        assert len(draws) > 1

    def test_handles_video_shorter_than_clip(self):
        idx = sample_indices(4, 16, stride=2, mode="tsn", training=False)
        assert len(idx) == 16
        assert idx.max() <= 3

    def test_single_frame_video(self):
        idx = sample_indices(1, 8, stride=1, mode="contiguous")
        assert np.all(idx == 0)

    def test_rejects_bad_arguments(self):
        with pytest.raises(ValueError):
            sample_indices(0, 8)
        with pytest.raises(ValueError):
            sample_indices(10, 0)


class TestWindowStarts:
    def test_covers_the_tail(self):
        starts = window_starts(100, span=32, hop=16)
        assert starts[0] == 0
        assert starts[-1] == 68  # 100 - 32, so the last frames are seen
        assert all(s + 32 <= 100 for s in starts)

    def test_short_video_yields_one_window(self):
        assert window_starts(10, span=32, hop=16) == [0]

    def test_no_duplicate_tail(self):
        starts = window_starts(96, span=32, hop=32)
        assert len(starts) == len(set(starts))
