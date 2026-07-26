"""Procedural synthetic footage.

This exists so the pipeline is *verifiable* without waiting on a real dataset: it
renders an articulated stick-dog whose motion signature differs per behaviour, and
emits it as real H.264 mp4s with span annotations.  A model that cannot learn this
has a bug; a model that can has a working data path, loss, sampler and decoder.

It is a test fixture and a demo, not training data — the appearance statistics of
a rendered stick figure have nothing to do with a real dog, so a model trained
here transfers to nothing.  Two deliberate choices keep it honest as a *test*
though:

* per-session randomised dog colour, size, background and camera jitter, so the
  only consistent cue is motion (a model that cheats on colour will fail);
* multi-segment sessions with span annotations, so span sampling, group-aware
  splitting and the audit path are all exercised end to end.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dataset import Annotation, save_annotations
from .labels import ACTIONS, POSTURE
from .video import write_video

# Locomotion speed (pixels/frame at 320px width), leg cadence, and how extended
# the legs are (1.0 = fully standing), which is what sets body height.
_POSTURE_MOTION: dict[str, tuple[float, float, float]] = {
    # posture: (forward speed, leg frequency, leg extension)
    "lying_down": (0.0, 0.0, 0.10),
    "sitting": (0.0, 0.0, 0.45),
    "standing": (0.0, 0.0, 1.00),
    "walking": (1.1, 0.18, 1.00),
    "trotting": (2.4, 0.34, 1.02),
    "running": (4.6, 0.55, 1.05),
}

# Paw line as a fraction of frame height — below the 0.62 horizon, so the dog
# stands on the ground rather than hovering in the sky.
_GROUND = 0.84


@dataclass
class DogStyle:
    """Per-session nuisance parameters — the things the model must ignore."""

    body_colour: tuple[int, int, int]
    ground_colour: tuple[int, int, int]
    sky_colour: tuple[int, int, int]
    scale: float
    camera_shake: float
    texture_seed: int

    @classmethod
    def random(cls, rng: np.random.Generator) -> "DogStyle":
        return cls(
            body_colour=tuple(int(v) for v in rng.integers(40, 230, size=3)),
            ground_colour=tuple(int(v) for v in rng.integers(60, 140, size=3)),
            sky_colour=tuple(int(v) for v in rng.integers(120, 235, size=3)),
            scale=float(rng.uniform(0.95, 1.45)),
            camera_shake=float(rng.uniform(0.0, 1.6)),
            texture_seed=int(rng.integers(0, 2**31 - 1)),
        )


class DogRenderer:
    """Draws the stick-dog for a given behaviour set and time index."""

    def __init__(self, width: int = 320, height: int = 240, style: DogStyle | None = None,
                 rng: np.random.Generator | None = None):
        self.width = width
        self.height = height
        self.rng = rng or np.random.default_rng()
        self.style = style or DogStyle.random(self.rng)
        self.x = width * 0.5
        self.direction = 1.0
        self._background = self._make_background()

    def _make_background(self) -> np.ndarray:
        rng = np.random.default_rng(self.style.texture_seed)
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        horizon = int(self.height * 0.62)
        canvas[:horizon] = self.style.sky_colour
        canvas[horizon:] = self.style.ground_colour
        # Low-frequency blotches so the background is not a flat colour (a flat
        # background makes frame-differencing unrealistically clean).
        for _ in range(28):
            cx, cy = rng.integers(0, self.width), rng.integers(horizon, self.height)
            radius = int(rng.integers(6, 30))
            shade = rng.integers(-26, 26)
            y0, y1 = max(0, cy - radius), min(self.height, cy + radius)
            x0, x1 = max(0, cx - radius), min(self.width, cx + radius)
            patch = canvas[y0:y1, x0:x1].astype(np.int16) + shade
            canvas[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)
        noise = rng.normal(0, 4, canvas.shape)
        return np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # -- pose ------------------------------------------------------------
    def _pose(self, posture: str, actions: set[str], t: int) -> dict:
        """Compute the articulated pose for one frame.

        Each behaviour contributes to a different degree of freedom, so a
        multi-label combination (``running`` + ``barking`` + ``tail_wagging``)
        produces a genuinely composite motion rather than one of the two.
        """
        speed, leg_frequency, extension = _POSTURE_MOTION[posture]
        scale = self.style.scale
        body_length = 62 * scale
        leg_length = 26 * scale * extension
        ground = self.height * _GROUND

        phase = t * leg_frequency * 2 * math.pi
        pose = {
            "cx": self.x,
            "cy": ground - leg_length - 5 * scale,
            "length": body_length,
            "thickness": max(2, int(9 * scale)),
            "leg_length": leg_length,
            "leg_phase": phase,
            "leg_amplitude": 13 * scale if leg_frequency else 2 * scale,
            "head_drop": 0.0,
            "head_radius": 11 * scale,
            "tail_angle": -0.5,
            "body_angle": 0.0,
            "front_leg_extra": 0.0,
            "hind_leg_extra": 0.0,
            "stretch": 1.0,
        }

        # Breathing keeps "static" postures from being literally frozen, so
        # alert_freeze remains distinguishable as the *only* motionless class.
        if "alert_freeze" not in actions:
            pose["cy"] += math.sin(t * 0.09) * 0.8 * scale
        if posture in ("walking", "trotting", "running"):
            pose["cy"] -= abs(math.sin(phase)) * (2.5 if posture == "running" else 1.2) * scale
        if posture == "sitting":
            pose["body_angle"] = 0.42
        if posture == "lying_down":
            pose["body_angle"] = 0.05
            pose["leg_amplitude"] = 1.0

        if "jumping" in actions:
            # Parabolic hops rather than a sine, so the vertical profile is
            # asymmetric like a real jump.
            period = 26.0
            u = (t % period) / period
            pose["cy"] -= 46 * scale * max(0.0, 4 * u * (1 - u))
        if "tail_wagging" in actions:
            pose["tail_angle"] = -0.5 + math.sin(t * 1.15) * 0.95  # fast, small mass
        if "barking" in actions:
            pose["head_radius"] *= 1.0 + 0.30 * max(0.0, math.sin(t * 0.85))
        if "eating_drinking" in actions:
            pose["head_drop"] = 22 * scale + math.sin(t * 0.55) * 3.5 * scale
        if "sniffing" in actions:
            pose["head_drop"] = 25 * scale
            pose["cx"] += math.sin(t * 0.16) * 7 * scale  # nose sweeping side to side
        if "digging" in actions:
            pose["head_drop"] = 18 * scale
            pose["front_leg_extra"] = math.sin(t * 1.5) * 15 * scale  # fast forepaws
        if "scratching" in actions:
            pose["hind_leg_extra"] = math.sin(t * 1.7) * 16 * scale
        if "shaking_off" in actions:
            pose["body_angle"] += math.sin(t * 2.1) * 0.30  # whole-body high frequency
            pose["cx"] += math.sin(t * 2.1) * 3.5 * scale
        if "rolling" in actions:
            pose["body_angle"] += (t * 0.22) % (2 * math.pi)
            pose["cy"] += 12 * scale
        if "stretching" in actions:
            pose["stretch"] = 1.0 + 0.42 * max(0.0, math.sin(t * 0.18))
        if "playing" in actions:
            # Erratic direction reversals: the signature of play is
            # unpredictability, not a single frequency.
            pose["cx"] += math.sin(t * 0.42) * 16 * scale + math.sin(t * 0.93) * 7 * scale
            pose["cy"] -= abs(math.sin(t * 0.5)) * 13 * scale
        if "alert_freeze" in actions:
            pose["tail_angle"] = -1.15
            pose["leg_amplitude"] = 0.0

        self.x += speed * self.direction * scale
        margin = body_length + 30
        if self.x > self.width - margin or self.x < margin:
            self.direction *= -1.0
            self.x = float(np.clip(self.x, margin, self.width - margin))
        return pose

    def frame(self, posture: str, actions: set[str], t: int) -> np.ndarray:
        import cv2

        pose = self._pose(posture, actions, t)
        canvas = self._background.copy()
        colour = self.style.body_colour
        shake = self.style.camera_shake
        jitter_x = self.rng.normal(0, shake)
        jitter_y = self.rng.normal(0, shake)

        cx = pose["cx"] + jitter_x
        cy = pose["cy"] + jitter_y
        length = pose["length"] * pose["stretch"]
        angle = pose["body_angle"]
        facing = self.direction
        thickness = pose["thickness"]

        def point(dx: float, dy: float) -> tuple[int, int]:
            rx = dx * math.cos(angle) - dy * math.sin(angle)
            ry = dx * math.sin(angle) + dy * math.cos(angle)
            return int(round(cx + rx * facing)), int(round(cy + ry))

        rear = point(-length / 2, 0)
        front = point(length / 2, 0)
        cv2.line(canvas, rear, front, colour, thickness, cv2.LINE_AA)

        head = point(length / 2 + 12 * self.style.scale, -6 * self.style.scale + pose["head_drop"])
        cv2.line(canvas, front, head, colour, max(2, thickness - 3), cv2.LINE_AA)
        cv2.circle(canvas, head, int(pose["head_radius"]), colour, -1, cv2.LINE_AA)
        # Ears: up when alert, otherwise flopped back.
        ear_dy = -14 if "alert_freeze" in actions else -6
        ear = point(length / 2 + 8 * self.style.scale, ear_dy + pose["head_drop"])
        cv2.line(canvas, head, ear, colour, 2, cv2.LINE_AA)

        tail_angle = pose["tail_angle"]
        tail = point(
            -length / 2 - 20 * self.style.scale * math.cos(tail_angle),
            -20 * self.style.scale * math.sin(tail_angle) - 4,
        )
        cv2.line(canvas, rear, tail, colour, 2, cv2.LINE_AA)

        amplitude = pose["leg_amplitude"]
        leg_length = pose["leg_length"]
        for index, (dx, extra) in enumerate(
            [
                (length * 0.34, pose["front_leg_extra"]),
                (length * 0.20, pose["front_leg_extra"] * 0.7),
                (-length * 0.20, pose["hind_leg_extra"] * 0.7),
                (-length * 0.34, pose["hind_leg_extra"]),
            ]
        ):
            swing = math.sin(pose["leg_phase"] + index * math.pi / 2) * amplitude
            hip = point(dx, 2)
            paw = point(dx + swing * 0.5 + extra, leg_length + abs(swing) * 0.25)
            cv2.line(canvas, hip, paw, colour, max(2, thickness - 4), cv2.LINE_AA)

        return canvas


# ---------------------------------------------------------------------------
# session generation
# ---------------------------------------------------------------------------
def _sample_behaviours(rng: np.random.Generator, postures: list[str], actions: list[str]) -> list[str]:
    """One posture plus a plausible number of compatible actions."""
    posture = str(rng.choice(postures))
    labels = [posture]
    incompatible = {
        "lying_down": {"jumping", "digging", "walking", "shaking_off"},
        "sitting": {"jumping", "rolling"},
        "running": {"eating_drinking", "sniffing", "digging", "scratching", "rolling", "alert_freeze"},
        "trotting": {"eating_drinking", "digging", "scratching", "rolling", "alert_freeze"},
        "walking": {"digging", "scratching", "rolling", "alert_freeze"},
        "standing": set(),
    }.get(posture, set())
    pool = [a for a in actions if a not in incompatible]
    n_actions = int(rng.choice([0, 1, 1, 2], p=[0.25, 0.4, 0.2, 0.15]))
    if pool and n_actions:
        labels += [str(a) for a in rng.choice(pool, size=min(n_actions, len(pool)), replace=False)]
    return labels


def generate_session(
    path: str | Path,
    rng: np.random.Generator,
    behaviours: list[str] | None = None,
    n_segments: int = 4,
    segment_seconds: float = 2.5,
    fps: float = 25.0,
    size: tuple[int, int] = (320, 240),
) -> tuple[Path, list[Annotation]]:
    """Render one multi-segment video plus its span annotations."""
    names = behaviours or list(POSTURE + ACTIONS)
    postures = [n for n in names if n in POSTURE] or list(POSTURE)
    actions = [n for n in names if n in ACTIONS]

    renderer = DogRenderer(width=size[0], height=size[1], rng=rng)
    frames_per_segment = max(4, int(round(segment_seconds * fps)))
    all_frames: list[np.ndarray] = []
    annotations: list[Annotation] = []
    group = Path(path).stem

    for segment in range(n_segments):
        labels = _sample_behaviours(rng, postures, actions)
        posture = labels[0]
        action_set = set(labels[1:])
        start_frame = len(all_frames)
        for t in range(frames_per_segment):
            all_frames.append(renderer.frame(posture, action_set, t))
        # Trim a frame at each end of the span: the transition frames genuinely
        # contain both behaviours, and labelling them as one is exactly the
        # boundary noise this pipeline is built to avoid.
        annotations.append(
            Annotation(
                video=str(path),
                labels=labels,
                start=(start_frame + 1) / fps,
                end=(len(all_frames) - 1) / fps,
                group=group,
                meta={"synthetic": True, "segment": segment},
            )
        )

    write_video(path, all_frames, fps=fps)
    return Path(path), annotations


def generate_dataset(
    root: str | Path,
    n_sessions: int = 24,
    behaviours: list[str] | None = None,
    segments_per_session: int = 4,
    segment_seconds: float = 2.5,
    fps: float = 25.0,
    size: tuple[int, int] = (320, 240),
    splits: dict[str, float] | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, Path]:
    """Render a full synthetic dataset with group-aware splits.

    Sessions are the group unit, so no session is ever split across train and val
    — the same discipline the real pipeline enforces.
    """
    from .dataset import make_splits

    root = Path(root)
    (root / "videos").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    everything: list[Annotation] = []

    for index in range(n_sessions):
        path = root / "videos" / f"session_{index:03d}.mp4"
        _, annotations = generate_session(
            path,
            rng=np.random.default_rng(rng.integers(0, 2**31 - 1)),
            behaviours=behaviours,
            n_segments=segments_per_session,
            segment_seconds=segment_seconds,
            fps=fps,
            size=size,
        )
        everything += annotations
        if verbose and (index + 1) % 5 == 0:
            print(f"  rendered {index + 1}/{n_sessions} sessions", flush=True)

    parts = make_splits(everything, splits or {"train": 0.75, "val": 0.25}, seed=seed)
    out: dict[str, Path] = {}
    for name, annotations in parts.items():
        # Store paths relative to the dataset root so the folder stays portable.
        for ann in annotations:
            ann.video = str(Path(ann.video).relative_to(root))
        out[name] = save_annotations(root / f"{name}.jsonl", annotations)
        if verbose:
            print(f"  {name}: {len(annotations)} spans -> {out[name]}")
    return out
