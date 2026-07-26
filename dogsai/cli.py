"""Command line interface: ``dogsai <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config
from .labels import DEFAULT_BEHAVIOURS, LabelSpace


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, default=None, help="JSON/YAML config file")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set train.epochs=60 --set model.preset=base",
    )


def _load_config(args: argparse.Namespace) -> Config:
    config = Config.load(args.config) if getattr(args, "config", None) else Config()
    overrides = list(getattr(args, "set", []) or [])
    for override in overrides:
        if "=" not in override:
            raise SystemExit(f"--set needs KEY=VALUE, got {override!r}")

    # A preset is a *base*, so it is resolved first and explicit --set overrides
    # are applied on top. Doing it the other way round means
    # `--set model.preset=nano --set data.image_size=112` silently trains at the
    # preset's resolution and ignores the 112 — a genuinely confusing failure.
    preset = next(
        (o.split("=", 1)[1].strip() for o in overrides if o.split("=", 1)[0].strip() == "model.preset"),
        None,
    )
    if preset is not None:
        config.model.preset = preset  # type: ignore[assignment]
        config.apply_preset()
    elif getattr(args, "config", None) is None:
        config.apply_preset()

    for override in overrides:
        key, raw = override.split("=", 1)
        _assign(config, key.strip(), raw.strip())

    if getattr(args, "data_root", None):
        config.data.root = args.data_root
    if getattr(args, "out_dir", None):
        config.train.out_dir = args.out_dir
    if getattr(args, "task", None):
        config.task = args.task
    return config


def _assign(config: Config, dotted: str, raw: str) -> None:
    target = config
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not hasattr(target, part):
            raise SystemExit(f"unknown config section {part!r} in {dotted!r}")
        target = getattr(target, part)
    leaf = parts[-1]
    if not hasattr(target, leaf):
        raise SystemExit(f"unknown config key {dotted!r}")
    current = getattr(target, leaf)
    setattr(target, leaf, _coerce(raw, current))


def _coerce(raw: str, current):
    if isinstance(current, bool):
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(current, int) and not isinstance(current, bool):
        return int(float(raw))
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, (list, tuple)):
        parts = [p for p in raw.replace(",", " ").split() if p]
        if isinstance(current, tuple):
            return tuple(float(p) for p in parts)
        return parts
    return raw


def _labels_from(config: Config, path: str | None) -> LabelSpace:
    if path:
        names = [n.strip() for n in Path(path).read_text().splitlines() if n.strip()]
        config.behaviours = names
    return LabelSpace.from_names(config.behaviours)


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_synth(args: argparse.Namespace) -> int:
    from .synth import generate_dataset

    print(f"rendering {args.sessions} synthetic sessions into {args.root} ...")
    generate_dataset(
        root=args.root,
        n_sessions=args.sessions,
        segments_per_session=args.segments,
        segment_seconds=args.segment_seconds,
        fps=args.fps,
        size=(args.width, args.height),
        seed=args.seed,
    )
    print(
        "done. next:\n"
        f"  dogsai audit --data-root {args.root}\n"
        f"  dogsai train --data-root {args.root} --set model.preset=nano --set train.epochs=12"
    )
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    from .dataset import make_splits, save_annotations
    from .importers import IMPORTERS, load_label_map

    importer = IMPORTERS[args.format]
    mapping = load_label_map(args.label_map)
    kwargs = {"label_map": mapping, "keep_unmapped": args.keep_unmapped}
    if args.format != "folders":
        kwargs["video_root"] = args.video_root
    result = importer(args.source, **kwargs)
    print(result.report())
    if not result.annotations:
        print("nothing imported.", file=sys.stderr)
        return 1

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.split:
        fractions = {"train": args.train_fraction, "val": 1.0 - args.train_fraction}
        parts = make_splits(result.annotations, fractions, seed=args.seed)
        for name, annotations in parts.items():
            path = save_annotations(out_root / f"{name}.jsonl", annotations)
            print(f"  wrote {len(annotations):5d} spans -> {path}")
    else:
        path = save_annotations(out_root / "all.jsonl", result.annotations)
        print(f"  wrote {len(result.annotations)} spans -> {path}")

    names = sorted({n for a in result.annotations for n in a.labels})
    (out_root / "behaviours.txt").write_text("\n".join(names) + "\n")
    print(f"  wrote {len(names)} behaviour names -> {out_root / 'behaviours.txt'}")
    print("\nnow run:  dogsai audit --data-root", out_root)
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    from .audit import audit_splits
    from .dataset import discover_split

    config = _load_config(args)
    labels = _labels_from(config, args.behaviours) if (args.behaviours or not args.any_label) else None

    splits = {}
    for name in args.splits:
        try:
            splits[name] = discover_split(config.data.root, name)
        except FileNotFoundError as exc:
            print(f"skipping {name}: {exc}")
    if not splits:
        print("no splits found.", file=sys.stderr)
        return 1

    clip_seconds = config.clip_span_frames / 30.0
    reports, cross = audit_splits(
        splits,
        labels,
        clip_seconds=clip_seconds,
        check_duplicates=not args.no_duplicate_check,
        cache_path=Path(config.data.root) / ".dogsai_meta.json",
    )
    failed = False
    for name, report in reports.items():
        print(f"\n### split: {name}")
        print(report.render())
        failed = failed or not report.ok
    if cross:
        print("\n### cross-split checks")
        for finding in cross:
            print(f"  {finding}")
        failed = failed or any(f.level == "error" for f in cross)
    else:
        print("\n### cross-split checks\n  no leakage detected between splits.")

    if args.json:
        payload = {
            "splits": {k: v.to_dict() for k, v in reports.items()},
            "cross_split": [
                {"level": f.level, "code": f.code, "message": f.message} for f in cross
            ],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    if failed:
        print("\naudit found errors that will corrupt or inflate training.")
    return 1 if (failed and args.strict) else 0


def cmd_train(args: argparse.Namespace) -> int:
    from .dataset import ClipDataset, discover_split
    from .engine import Trainer

    config = _load_config(args)
    labels = _labels_from(config, args.behaviours)
    root = Path(config.data.root)
    cache = root / ".dogsai_meta.json"

    train_annotations = discover_split(root, config.data.train_split)
    val_annotations = None
    try:
        val_annotations = discover_split(root, config.data.val_split)
    except FileNotFoundError:
        print(f"no {config.data.val_split} split found — training without validation")

    if args.cache:
        # Decode once, then train off a memmapped uint8 array. On real footage
        # this is the difference between minutes and seconds per epoch.
        from .cache import CacheSpec, CachedClipDataset, build_split_caches, estimate_cache_size

        spec = CacheSpec(frames=args.cache_frames, size=args.cache_size)
        total = len(train_annotations) + (len(val_annotations) if val_annotations else 0)
        print(f"clip cache: {spec.frames} frames at {spec.size}px, "
              f"~{estimate_cache_size(total, spec):.2f} GB total")
        splits = {config.data.train_split: train_annotations}
        if val_annotations:
            splits[config.data.val_split] = val_annotations
        dirs = build_split_caches(
            splits, root, spec, workers=max(1, config.data.num_workers), force=args.rebuild_cache
        )
        train_set = CachedClipDataset(
            dirs[config.data.train_split], labels, config.data, config.task,
            training=True, seed=config.train.seed,
        )
        val_set = (
            CachedClipDataset(
                dirs[config.data.val_split], labels, config.data, config.task,
                training=False, seed=config.train.seed,
            )
            if val_annotations
            else None
        )
    else:
        train_set = ClipDataset(
            train_annotations, labels, config.data, config.task,
            training=True, cache_path=cache, seed=config.train.seed,
        )
        val_set = (
            ClipDataset(
                val_annotations, labels, config.data, config.task,
                training=False, cache_path=cache, seed=config.train.seed,
            )
            if val_annotations
            else None
        )

    for dataset, name in ((train_set, "train"), (val_set, "val")):
        if dataset and dataset.skipped:
            print(f"warning: skipped {len(dataset.skipped)} unusable {name} annotations "
                  f"(first: {dataset.skipped[0][0]} — {dataset.skipped[0][1]})")

    trainer = Trainer(config, train_set, val_set, labels, device=args.device)
    if args.resume:
        trainer.resume(args.resume, weights_only=args.resume_weights_only)
    trainer.fit()

    result = trainer.evaluate()
    if result is not None:
        print("\nper-class results (best EMA weights, tuned thresholds):")
        print(result.table(sort_by="ap" if config.task == "multilabel" else "recall"))
    print(f"\ncheckpoints in {config.train.out_dir}/  (best.pt, last.pt)")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .dataset import ClipDataset, discover_split
    from .engine import Trainer, load_checkpoint
    from .metrics import evaluate, format_confusion

    model, config, labels, extra = load_checkpoint(args.checkpoint, device=args.device or "cpu")
    if args.data_root:
        config.data.root = args.data_root
    root = Path(config.data.root)
    annotations = discover_split(root, args.split)
    dataset = ClipDataset(
        annotations, labels, config.data, config.task,
        training=False, cache_path=root / ".dogsai_meta.json",
    )
    print(f"evaluating {len(dataset)} clips from {args.split} "
          f"({'EMA' if extra['used_ema'] else 'raw'} weights, epoch {extra['epoch']})")

    trainer = Trainer.__new__(Trainer)  # reuse the batched predict path only
    trainer.config = config
    trainer.device = next(model.parameters()).device
    trainer.model = model
    trainer.ema = None
    trainer.use_amp = False
    loader = Trainer._make_loader(trainer, dataset, training=False)
    logits, targets = Trainer.predict_loader(trainer, loader, use_ema=False)

    saved = extra.get("thresholds")
    result = evaluate(
        logits, targets, labels.to_list(), task=config.task,
        **({"thresholds": saved, "tune": False} if (saved and config.task == "multilabel" and not args.tune) else {}),
    )
    print("\n" + result.summary())
    print("\n" + result.table(sort_by="ap" if config.task == "multilabel" else "recall"))
    if result.confusion is not None and args.confusion:
        print("\nconfusion (row-normalised, rows=truth):")
        print(format_confusion(result.confusion, labels.to_list()))
    if args.json:
        Path(args.json).write_text(json.dumps(result.to_dict(), indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .predict import BehaviourPredictor, render_overlay
    from .video import find_videos

    predictor = BehaviourPredictor(args.checkpoint, device=args.device)
    if args.threshold is not None:
        predictor.thresholds[:] = args.threshold
    if args.window_stride is not None:
        predictor.config.infer.window_stride = args.window_stride
    if args.smooth is not None:
        predictor.config.infer.smooth = args.smooth
    predictor.config.infer.tta_hflip = args.tta

    videos = []
    for target in args.videos:
        videos += [str(p) for p in find_videos(target)]
    if not videos:
        print("no videos found.", file=sys.stderr)
        return 1
    print(f"{len(videos)} video(s), {'EMA' if predictor.extra['used_ema'] else 'raw'} weights\n")

    results = []
    for video in videos:
        prediction = predictor.predict(video)
        results.append(prediction.to_dict())
        print(f"=== {video}")
        print(prediction.timeline())
        print()
        for span in prediction.spans:
            print(f"  {span}")
        totals = prediction.time_per_behaviour()
        if totals:
            print("\n  time per behaviour: "
                  + ", ".join(f"{k} {v:.1f}s" for k, v in totals.items()))
            print(f"  dominant: {prediction.dominant()}   "
                  f"arousal-flagged: {prediction.arousal_fraction() * 100:.0f}% of runtime")
        if not args.no_feeling:
            print()
            print("\n".join("  " + line for line in prediction.affect().render().splitlines()))
        if args.explain is not None:
            explanation = predictor.explain(video, at=args.explain)
            print(f"\n  at {args.explain:.1f}s (window {explanation['window'][0]:.2f}-"
                  f"{explanation['window'][1]:.2f}s):")
            for item in explanation["top"]:
                print(f"    {item['behaviour']:<18} {item['score']:.3f}")
            weights = explanation["frame_attention"]
            bars = "".join("▁▂▃▄▅▆▇█"[min(7, int(w * len(weights) * 4))] for w in weights)
            print(f"    frame attention: {bars}")
        if args.overlay:
            out = Path(args.overlay)
            path = out / f"{Path(video).stem}_annotated.mp4" if len(videos) > 1 or out.is_dir() else out
            render_overlay(prediction, path)
            print(f"\n  overlay -> {path}")
        print()

    if args.json:
        Path(args.json).write_text(json.dumps(results if len(results) > 1 else results[0], indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    from .datasets_hub import REGISTRY, describe_registry, download, prepare

    if not args.dataset:
        print("available datasets\n" + "=" * 60)
        print(describe_registry())
        print("\nfetch one with:  dogsai fetch dogbehaviour")
        return 0
    if args.dataset not in REGISTRY:
        print(f"unknown dataset {args.dataset!r}; known: {', '.join(REGISTRY)}",
              file=sys.stderr)
        return 1

    spec = REGISTRY[args.dataset]
    print(spec.summary())
    if not args.skip_download:
        print(f"\ndownloading ~{spec.approx_gb:.1f} GB into {args.raw_root} ...")
        download(args.dataset, args.raw_root, workers=args.workers)
    print(f"\nconverting annotations -> {args.out}")
    written = prepare(
        args.dataset,
        args.raw_root,
        args.out,
        fractions={"train": args.train_fraction, "val": 1.0 - args.train_fraction},
        seed=args.seed,
    )
    for name, path in written.items():
        print(f"  {name:<12} {path}")
    print(f"\nnow run:\n  dogsai audit --data-root {args.out}\n"
          f"  dogsai train --data-root {args.out} "
          f"--behaviours {args.out}/behaviours.txt")
    return 0


def cmd_feeling(args: argparse.Namespace) -> int:
    from .predict import BehaviourPredictor

    predictor = BehaviourPredictor(args.checkpoint, device=args.device)
    prediction = predictor.predict(args.video)
    reading = prediction.affect()
    print(f"=== {args.video}  ({prediction.meta.duration:.1f}s)\n")
    print(prediction.timeline())
    print()
    print(reading.render())
    if args.json:
        Path(args.json).write_text(
            json.dumps({**prediction.to_dict(), "affect": reading.to_dict()}, indent=2) + "\n"
        )
        print(f"\nwrote {args.json}")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from .export import benchmark

    if args.checkpoint:
        stats = benchmark(checkpoint=args.checkpoint, batch=args.batch,
                          iterations=args.iterations, device=args.device or "cpu")
        print(f"checkpoint: {args.checkpoint}")
    else:
        config = _load_config(args)
        stats = benchmark(config=config, num_classes=len(config.behaviours), batch=args.batch,
                          iterations=args.iterations, device=args.device or "cpu")
        print(f"preset: {config.model.preset}  "
              f"input: {config.data.clip_frames}x{config.data.image_size}x{config.data.image_size}")
    print(f"  parameters      : {stats['parameters_m']:.2f} M")
    print(f"  MACs / clip     : {stats['macs_g']:.2f} G")
    print(f"  latency         : {stats['latency_ms']:.1f} ms  (batch {stats['batch']}, {stats['device']})")
    print(f"  throughput      : {stats['clips_per_second']:.1f} clips/s")
    print(f"  realtime factor : {stats['realtime_factor']:.1f}x  (30fps source video)")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .export import export_onnx, export_torchscript

    if args.format == "torchscript":
        path = export_torchscript(args.checkpoint, args.out)
    else:
        path = export_onnx(args.checkpoint, args.out, opset=args.opset)
    print(f"wrote {path}")
    print(f"wrote {Path(path).with_suffix('.meta.json')}  (labels, normalisation, thresholds)")
    return 0


def cmd_behaviours(args: argparse.Namespace) -> int:
    from .labels import ACTIONS, AROUSAL_FLAGS, POSTURE

    print(f"default taxonomy ({len(DEFAULT_BEHAVIOURS)} behaviours)\n")
    print("posture / locomotion (mutually exclusive):")
    for name in POSTURE:
        print(f"  {name}")
    print("\nactions (can co-occur):")
    for name in ACTIONS:
        flag = "  [arousal-flagged]" if name in AROUSAL_FLAGS else ""
        print(f"  {name}{flag}")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dogsai",
        description="Dog behaviour recognition from video — train, audit, and run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "typical flow:\n"
            "  dogsai synth --root data/synth               # make a runnable dataset\n"
            "  dogsai prepare --format label-studio ...     # or import your own\n"
            "  dogsai audit --data-root data/synth          # check labels before training\n"
            "  dogsai train --data-root data/synth\n"
            "  dogsai predict runs/dognet/best.pt clip.mp4\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    synth = subparsers.add_parser("synth", help="render a synthetic dataset for testing")
    synth.add_argument("--root", default="data/synth")
    synth.add_argument("--sessions", type=int, default=24)
    synth.add_argument("--segments", type=int, default=4)
    synth.add_argument("--segment-seconds", type=float, default=2.5)
    synth.add_argument("--fps", type=float, default=25.0)
    synth.add_argument("--width", type=int, default=320)
    synth.add_argument("--height", type=int, default=240)
    synth.add_argument("--seed", type=int, default=0)
    synth.set_defaults(func=cmd_synth)

    prepare = subparsers.add_parser("prepare", help="import annotations into span format")
    prepare.add_argument("source", help="export file, or dataset root for --format folders")
    prepare.add_argument("--format", required=True,
                         choices=["label-studio", "cvat", "csv", "ava", "folders"])
    prepare.add_argument("--out", default="data/prepared")
    prepare.add_argument("--video-root", default=None, help="where the mp4 files live")
    prepare.add_argument("--label-map", default=None, help='JSON {"source": "target"} mapping')
    prepare.add_argument("--keep-unmapped", action="store_true",
                         help="keep labels absent from --label-map instead of dropping them")
    prepare.add_argument("--split", action="store_true", help="also write train/val splits")
    prepare.add_argument("--train-fraction", type=float, default=0.8)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.set_defaults(func=cmd_prepare)

    audit = subparsers.add_parser("audit", help="validate a dataset before training")
    _add_common(audit)
    audit.add_argument("--data-root", default=None)
    audit.add_argument("--splits", nargs="+", default=["train", "val"])
    audit.add_argument("--behaviours", default=None, help="text file, one behaviour per line")
    audit.add_argument("--any-label", action="store_true",
                       help="do not check labels against a fixed taxonomy")
    audit.add_argument("--no-duplicate-check", action="store_true",
                       help="skip perceptual near-duplicate hashing (faster)")
    audit.add_argument("--json", default=None)
    audit.add_argument("--strict", action="store_true", help="exit non-zero on errors")
    audit.set_defaults(func=cmd_audit)

    train = subparsers.add_parser("train", help="train a model")
    _add_common(train)
    train.add_argument("--data-root", default=None)
    train.add_argument("--out-dir", default=None)
    train.add_argument("--task", choices=["multiclass", "multilabel"], default=None)
    train.add_argument("--behaviours", default=None, help="text file, one behaviour per line")
    train.add_argument("--device", default=None)
    train.add_argument("--cache", action="store_true",
                       help="decode every clip once into a memmapped array first; "
                            "usually a large speedup, at some augmentation diversity")
    train.add_argument("--cache-frames", type=int, default=32,
                       help="frames stored per clip; must exceed data.clip_frames "
                            "so temporal jitter survives")
    train.add_argument("--cache-size", type=int, default=176,
                       help="cached resolution; must exceed data.image_size so "
                            "random-resized-crop still has pixels to pick")
    train.add_argument("--rebuild-cache", action="store_true")
    train.add_argument("--resume", default=None, metavar="CHECKPOINT",
                       help="continue a run: restores weights, optimiser, EMA and epoch")
    train.add_argument("--resume-weights-only", action="store_true",
                       help="load weights but restart the optimiser and schedule "
                            "(for fine-tuning onto other data)")
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser("eval", help="evaluate a checkpoint on a split")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("--data-root", default=None)
    evaluate.add_argument("--split", default="val")
    evaluate.add_argument("--device", default=None)
    evaluate.add_argument("--tune", action="store_true",
                          help="refit thresholds on this split (optimistic; for analysis only)")
    evaluate.add_argument("--confusion", action="store_true")
    evaluate.add_argument("--json", default=None)
    evaluate.set_defaults(func=cmd_eval)

    predict = subparsers.add_parser("predict", help="annotate videos with a trained model")
    predict.add_argument("checkpoint")
    predict.add_argument("videos", nargs="+", help="mp4 files or directories")
    predict.add_argument("--device", default=None)
    predict.add_argument("--threshold", type=float, default=None,
                         help="override the per-class tuned thresholds")
    predict.add_argument("--window-stride", type=float, default=None,
                         help="window hop as a fraction of clip length (0.25 = denser)")
    predict.add_argument("--smooth", type=int, default=None, help="median filter width in windows")
    predict.add_argument("--tta", action="store_true", help="average with the horizontal flip")
    predict.add_argument("--overlay", default=None, help="write annotated mp4 here")
    predict.add_argument("--explain", type=float, default=None, metavar="SECONDS",
                         help="show frame attention for the window at this timestamp")
    predict.add_argument("--no-feeling", action="store_true",
                         help="skip the body-language / affect read")
    predict.add_argument("--json", default=None)
    predict.set_defaults(func=cmd_predict)

    fetch = subparsers.add_parser("fetch", help="download and prepare a real public dataset")
    fetch.add_argument("dataset", nargs="?", default=None,
                       help="registry key; omit to list what is available")
    fetch.add_argument("--raw-root", default="dogsai_data/raw")
    fetch.add_argument("--out", default="dogsai_data/prepared")
    fetch.add_argument("--workers", type=int, default=8)
    fetch.add_argument("--train-fraction", type=float, default=0.8)
    fetch.add_argument("--seed", type=int, default=0)
    fetch.add_argument("--skip-download", action="store_true",
                       help="only convert an already-downloaded copy")
    fetch.set_defaults(func=cmd_fetch)

    feeling = subparsers.add_parser(
        "feeling",
        help="read one video's body language: arousal, valence and the evidence",
    )
    feeling.add_argument("checkpoint")
    feeling.add_argument("video")
    feeling.add_argument("--device", default=None)
    feeling.add_argument("--json", default=None)
    feeling.set_defaults(func=cmd_feeling)

    info = subparsers.add_parser("info", help="model size, MACs and measured latency")
    _add_common(info)
    info.add_argument("--checkpoint", default=None)
    info.add_argument("--batch", type=int, default=1)
    info.add_argument("--iterations", type=int, default=20)
    info.add_argument("--device", default=None)
    info.set_defaults(func=cmd_info)

    export = subparsers.add_parser("export", help="export to TorchScript or ONNX")
    export.add_argument("checkpoint")
    export.add_argument("out")
    export.add_argument("--format", choices=["torchscript", "onnx"], default="torchscript")
    export.add_argument("--opset", type=int, default=17)
    export.set_defaults(func=cmd_export)

    behaviours = subparsers.add_parser("behaviours", help="list the default taxonomy")
    behaviours.set_defaults(func=cmd_behaviours)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return 130
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
