"""Video decoding, clip sampling and encoding.

Decoding is the bottleneck of every video pipeline, so this module is written
around three rules:

1. **Never decode a frame you will not use.**  We decode exactly the frames the
   sampler asked for, seeking to the nearest keyframe first.
2. **Resize inside the decoder.**  swscale downscaling during ``reformat`` is far
   cheaper than decoding at full resolution and resizing in Python, and it cuts
   peak memory by an order of magnitude on 1080p input.
3. **Probe once.**  Frame counts are cached, because counting frames by decoding
   an entire file is the classic way to make epoch 1 take an hour.

Backends: PyAV (preferred — real seeking, no global state) with an OpenCV
fallback.  Both ship prebuilt ffmpeg, so neither needs a system install.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np

Backend = Literal["auto", "av", "cv2"]

VIDEO_SUFFIXES: frozenset[str] = frozenset(
    {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".mpg", ".mpeg"}
)

# Decoding sequentially from the start beats seeking for short prefixes: a seek
# lands on a keyframe and often has to decode most of the way back anyway.
_SEEK_THRESHOLD = 90


class VideoError(RuntimeError):
    """Raised when a file cannot be opened or yields no usable frames."""


@dataclass(frozen=True)
class VideoMeta:
    path: str
    n_frames: int
    fps: float
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else 0.0

    def time_of(self, frame_index: int) -> float:
        return frame_index / self.fps if self.fps > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "n_frames": self.n_frames,
            "fps": round(self.fps, 6),
            "width": self.width,
            "height": self.height,
        }


def _have_av() -> bool:
    try:
        import av  # noqa: F401

        return True
    except Exception:
        return False


def _resolve_backend(backend: Backend) -> str:
    if backend == "auto":
        return "av" if _have_av() else "cv2"
    if backend == "av" and not _have_av():
        raise VideoError("PyAV backend requested but `av` is not installed")
    return backend


# ---------------------------------------------------------------------------
# probing
# ---------------------------------------------------------------------------
def probe(path: str | Path, backend: Backend = "auto") -> VideoMeta:
    """Read stream metadata without decoding the whole file."""
    path = str(path)
    if not Path(path).exists():
        raise VideoError(f"no such video: {path}")
    if _resolve_backend(backend) == "av":
        return _probe_av(path)
    return _probe_cv2(path)


def _probe_av(path: str) -> VideoMeta:
    import av

    try:
        with av.open(path) as container:
            if not container.streams.video:
                raise VideoError(f"no video stream in {path}")
            stream = container.streams.video[0]
            fps = float(stream.average_rate or stream.guessed_rate or 0) or 30.0
            n = int(stream.frames or 0)
            if n <= 0:
                # Container did not store a frame count (common for streamed or
                # remuxed files) — derive it from duration instead of decoding.
                dur = stream.duration
                tb = stream.time_base
                if dur and tb:
                    n = int(round(float(dur * tb) * fps))
                elif container.duration:
                    n = int(round(container.duration / 1_000_000 * fps))
            width = int(stream.codec_context.width or 0)
            height = int(stream.codec_context.height or 0)
    except VideoError:
        raise
    except Exception as exc:
        raise VideoError(f"failed to probe {path}: {exc}") from exc
    if n <= 0:
        n = _count_frames_av(path)
    if n <= 0:
        raise VideoError(f"{path} appears to contain no frames")
    return VideoMeta(path, n, fps, width, height)


def _count_frames_av(path: str) -> int:
    """Last resort: decode packets (not frames) to count. Cheap-ish, exact enough."""
    import av

    count = 0
    with av.open(path) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            if packet.dts is None:
                continue
            count += packet.decode().__len__()
    return count


def _probe_cv2(path: str) -> VideoMeta:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoError(f"cannot open {path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if n <= 0:
        raise VideoError(f"{path} reports {n} frames")
    return VideoMeta(path, n, fps, width, height)


class MetaCache:
    """JSON-backed cache of :class:`VideoMeta`, keyed by path + mtime + size.

    Invalidation is automatic: if a file is re-encoded its mtime/size change and
    the entry is recomputed.
    """

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._entries: dict[str, dict] = {}
        self._dirty = False
        if self.path and self.path.exists():
            try:
                self._entries = json.loads(self.path.read_text())
            except Exception:
                self._entries = {}

    @staticmethod
    def _stamp(p: Path) -> str:
        st = p.stat()
        return f"{int(st.st_mtime)}:{st.st_size}"

    def get(self, video: str | Path, backend: Backend = "auto") -> VideoMeta:
        p = Path(video)
        key = str(p)
        stamp = self._stamp(p)
        hit = self._entries.get(key)
        if hit and hit.get("stamp") == stamp:
            return VideoMeta(
                key, hit["n_frames"], hit["fps"], hit["width"], hit["height"]
            )
        meta = probe(p, backend=backend)
        self._entries[key] = {**meta.to_dict(), "stamp": stamp}
        self._dirty = True
        return meta

    def flush(self) -> None:
        if self.path and self._dirty:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._entries))
            tmp.replace(self.path)
            self._dirty = False


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------
def read_frames(
    path: str | Path,
    indices: Sequence[int],
    size: tuple[int, int] | None = None,
    backend: Backend = "auto",
) -> np.ndarray:
    """Decode the requested frame indices as ``uint8`` RGB ``(T, H, W, 3)``.

    ``indices`` need not be sorted or unique, and out-of-range values are
    clamped, so a sampler can ask for ``[0, 0, 0, 1, 2]`` on a 3-frame video and
    still get a well-formed clip.  ``size`` is ``(width, height)`` and is applied
    during decode.
    """
    path = str(path)
    if len(indices) == 0:
        raise ValueError("no frame indices requested")
    order = np.asarray(indices, dtype=np.int64)
    wanted = np.unique(order)
    resolved = _resolve_backend(backend)
    if resolved == "av":
        frames = _decode_av(path, wanted, size)
    else:
        frames = _decode_cv2(path, wanted, size)
    lut = {int(idx): frame for idx, frame in zip(wanted, frames)}
    return np.stack([lut[int(i)] for i in order])


def _decode_av(
    path: str, wanted: np.ndarray, size: tuple[int, int] | None
) -> list[np.ndarray]:
    import av

    out: list[np.ndarray] = []
    kwargs = {}
    if size is not None:
        kwargs = {"width": size[0], "height": size[1], "format": "rgb24"}
    else:
        kwargs = {"format": "rgb24"}

    try:
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"  # frame+slice threading
            fps = float(stream.average_rate or stream.guessed_rate or 30.0)
            time_base = stream.time_base
            start = stream.start_time or 0

            first = int(wanted[0])
            if first > _SEEK_THRESHOLD and time_base:
                # Seek backwards to the keyframe at/just before the target.
                target_pts = int(first / fps / float(time_base)) + start
                try:
                    container.seek(target_pts, stream=stream, backward=True)
                except Exception:
                    pass  # unseekable input: fall through to sequential decode

            cursor = 0  # index into `wanted`
            fallback_counter = 0
            last: np.ndarray | None = None
            for frame in container.decode(stream):
                if frame.pts is not None and time_base:
                    idx = int(round(float((frame.pts - start) * time_base) * fps))
                else:
                    idx = fallback_counter
                fallback_counter = idx + 1

                array: np.ndarray | None = None
                # Monotonic assignment: everything still owed that is <= this
                # decoded frame gets this frame.  Handles VFR, dropped frames
                # and duplicate requests in one pass.
                while cursor < len(wanted) and int(wanted[cursor]) <= idx:
                    if array is None:
                        array = frame.reformat(**kwargs).to_ndarray()
                    out.append(array)
                    cursor += 1
                if array is not None:
                    last = array
                if cursor >= len(wanted):
                    break

            if cursor < len(wanted):
                # Requests past the end of the stream (a truncated file, or a
                # frame count the container lied about): clamp to the last frame.
                if last is None:
                    last = _decode_first_frame_av(path, kwargs)
                out.extend([last] * (len(wanted) - cursor))
    except VideoError:
        raise
    except Exception as exc:
        raise VideoError(f"failed to decode {path}: {exc}") from exc

    if not out:
        raise VideoError(f"decoded zero frames from {path}")
    return out


def _decode_first_frame_av(path: str, kwargs: dict) -> np.ndarray:
    import av

    with av.open(path) as container:
        for frame in container.decode(container.streams.video[0]):
            return frame.reformat(**kwargs).to_ndarray()
    raise VideoError(f"{path} has no decodable frames")


def _decode_cv2(
    path: str, wanted: np.ndarray, size: tuple[int, int] | None
) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise VideoError(f"cannot open {path}")
    out: list[np.ndarray] = []
    try:
        first = int(wanted[0])
        base = 0
        if first > _SEEK_THRESHOLD:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(first))
            base = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        cursor = 0
        idx = base - 1
        last: np.ndarray | None = None
        while cursor < len(wanted):
            # grab() skips decode work for frames we do not want.
            if not cap.grab():
                break
            idx += 1
            if int(wanted[cursor]) > idx:
                continue
            ok, bgr = cap.retrieve()
            if not ok:
                break
            if size is not None:
                bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            while cursor < len(wanted) and int(wanted[cursor]) <= idx:
                out.append(rgb)
                cursor += 1
            last = rgb
        if cursor < len(wanted):
            if last is None:
                raise VideoError(f"decoded zero frames from {path}")
            out.extend([last] * (len(wanted) - cursor))
    finally:
        cap.release()
    return out


# ---------------------------------------------------------------------------
# clip sampling
# ---------------------------------------------------------------------------
def sample_indices(
    n_frames: int,
    clip_frames: int,
    stride: int = 1,
    mode: Literal["tsn", "contiguous"] = "tsn",
    training: bool = False,
    rng: np.random.Generator | None = None,
    start: int | None = None,
) -> np.ndarray:
    """Pick ``clip_frames`` frame indices out of ``n_frames``.

    ``tsn`` divides the window into equal segments and takes one frame from each
    (jittered while training, centred otherwise) — this covers slow behaviours
    without a longer clip.  ``contiguous`` takes a dense run at ``stride``, which
    preserves fine motion detail.
    """
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    if clip_frames <= 0:
        raise ValueError("clip_frames must be positive")
    rng = rng or np.random.default_rng()
    span = clip_frames * max(1, stride)

    if mode == "contiguous":
        if start is None:
            slack = max(0, n_frames - span)
            begin = int(rng.integers(0, slack + 1)) if training and slack else slack // 2
        else:
            begin = start
        idx = begin + np.arange(clip_frames, dtype=np.int64) * max(1, stride)
        return np.clip(idx, 0, n_frames - 1)

    # TSN-style segment sampling.
    window = min(span, n_frames) if start is None else span
    begin = 0
    if start is not None:
        begin = start
    elif n_frames > window:
        slack = n_frames - window
        begin = int(rng.integers(0, slack + 1)) if training else slack // 2

    edges = np.linspace(0, window, clip_frames + 1)
    if training:
        offsets = rng.random(clip_frames) * np.diff(edges)
    else:
        offsets = np.diff(edges) * 0.5
    idx = np.floor(begin + edges[:-1] + offsets).astype(np.int64)
    return np.clip(idx, 0, n_frames - 1)


def window_starts(n_frames: int, span: int, hop: int) -> list[int]:
    """Sliding-window start frames covering a whole video, tail included."""
    span = max(1, span)
    hop = max(1, hop)
    if n_frames <= span:
        return [0]
    starts = list(range(0, n_frames - span + 1, hop))
    if starts[-1] != n_frames - span:
        starts.append(n_frames - span)
    return starts


def find_videos(root: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(root)
    if root.is_file():
        return [root]
    it: Iterable[Path] = root.rglob("*") if recursive else root.glob("*")
    return sorted(p for p in it if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)


# ---------------------------------------------------------------------------
# encoding (used by the synthetic-data generator and the overlay renderer)
# ---------------------------------------------------------------------------
def preferred_encoder() -> str:
    """Pick the best encoder actually compiled into the installed PyAV."""
    import av

    for name in ("libx264", "h264", "mpeg4"):
        try:
            av.codec.Codec(name, "w")
            return name
        except Exception:
            continue
    raise VideoError("no usable video encoder found in this PyAV build")


def write_video(
    path: str | Path,
    frames: Iterable[np.ndarray],
    fps: float = 30.0,
    crf: int = 23,
) -> Path:
    """Encode ``uint8`` RGB frames to an mp4."""
    import av

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    codec = preferred_encoder()
    container = av.open(str(path), mode="w")
    stream = None
    try:
        for frame in frames:
            array = np.ascontiguousarray(frame.astype(np.uint8, copy=False))
            if stream is None:
                h, w = array.shape[:2]
                # H.264 needs even dimensions for yuv420p.
                stream = container.add_stream(codec, rate=int(round(fps)))
                stream.width = w - (w % 2)
                stream.height = h - (h % 2)
                stream.pix_fmt = "yuv420p"
                if codec == "libx264":
                    stream.options = {"crf": str(crf), "preset": "veryfast"}
            array = array[: stream.height, : stream.width]
            av_frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)
        if stream is None:
            raise VideoError("write_video received no frames")
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    return path


def frames_to_time(n_frames: int, fps: float) -> float:
    return n_frames / fps if fps > 0 else 0.0


def seconds_to_frames(seconds: float, fps: float) -> int:
    return int(math.ceil(seconds * fps)) if fps > 0 else 0
