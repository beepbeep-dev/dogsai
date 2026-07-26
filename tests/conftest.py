"""Shared fixtures.

Videos are rendered once per session (encoding is the slow part) into a temp dir
that is reused by every test that needs real decodable footage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dogsai.synth import DogRenderer, generate_dataset, generate_session
from dogsai.video import write_video


@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory) -> Path:
    """A 60-frame 96x72 video whose frame index is encoded in its pixels.

    Frame ``i`` has a red channel of ``i * 4``, which lets decoder tests assert
    that they got *the frames they asked for* rather than merely 'some frames'.
    """
    path = tmp_path_factory.mktemp("video") / "counter.mp4"
    frames = []
    for i in range(60):
        frame = np.zeros((72, 96, 3), dtype=np.uint8)
        frame[:, :, 0] = min(255, i * 4)
        frame[:, :, 1] = 128
        frame[i % 72, :, 2] = 255  # a moving line, so motion is non-trivial
        frames.append(frame)
    write_video(path, frames, fps=30.0, crf=0)
    return path


@pytest.fixture(scope="session")
def session_video(tmp_path_factory) -> tuple[Path, list]:
    path = tmp_path_factory.mktemp("session") / "session.mp4"
    video, annotations = generate_session(
        path,
        rng=np.random.default_rng(0),
        n_segments=3,
        segment_seconds=1.5,
        fps=25.0,
        size=(160, 120),
    )
    return video, annotations


@pytest.fixture(scope="session")
def mini_dataset(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("dataset")
    generate_dataset(
        root,
        n_sessions=6,
        segments_per_session=2,
        segment_seconds=1.2,
        fps=25.0,
        size=(160, 120),
        seed=3,
        verbose=False,
    )
    return root


@pytest.fixture
def renderer() -> DogRenderer:
    return DogRenderer(width=160, height=120, rng=np.random.default_rng(1))
