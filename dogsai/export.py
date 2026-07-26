"""Export a trained checkpoint to TorchScript or ONNX.

Exported graphs are self-contained on purpose: the motion stem lives *inside* the
model, so a deployment target feeds normalised RGB clips and needs no knowledge of
frame differencing.  The per-class thresholds are written alongside as JSON, since
neither format has anywhere sensible to keep them and shipping the graph without
them silently reverts a tuned model to a 0.5 cut.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .engine import load_checkpoint


def _example_input(config, batch: int = 1) -> torch.Tensor:
    return torch.randn(
        batch, 3, config.data.clip_frames, config.data.image_size, config.data.image_size
    )


def _sidecar(path: Path, config, labels, extra: dict) -> Path:
    meta = {
        "behaviours": labels.to_list(),
        "task": config.task,
        "input": {
            "layout": "NCTHW",
            "clip_frames": config.data.clip_frames,
            "image_size": config.data.image_size,
            "frame_stride": config.data.frame_stride,
            "normalisation": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        },
        "thresholds": extra.get("thresholds"),
        "activation": "softmax" if config.task == "multiclass" else "sigmoid",
        "trained_epochs": extra.get("epoch"),
    }
    out = path.with_suffix(".meta.json")
    out.write_text(json.dumps(meta, indent=2) + "\n")
    return out


def export_torchscript(
    checkpoint: str | Path, out_path: str | Path, freeze: bool = True
) -> Path:
    """Trace, freeze and save, then verify the reloaded graph matches eager.

    Note what is deliberately *not* done here: ``torch.jit.optimize_for_inference``.
    It produces a graph this PyTorch cannot deserialise ("required keyword
    attribute 'value' is undefined"), so an optimised export saves fine and then
    fails at load — the worst possible failure mode. Freezing alone is safe and
    round-trips exactly; a deployment can call ``optimize_for_inference`` on the
    module it just loaded, where the result never has to survive serialisation.
    """
    model, config, labels, extra = load_checkpoint(checkpoint, device="cpu")
    model.eval()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    example = _example_input(config)
    with torch.no_grad():
        # Trace rather than script: the forward path is static, and tracing keeps
        # the graph free of the return_attention branch.
        traced = torch.jit.trace(model, example, strict=False)
        if freeze:
            traced = torch.jit.freeze(traced)
        traced.save(str(out_path))

        reference = model(example)
        replayed = torch.jit.load(str(out_path))(example)
    drift = float((reference - replayed).abs().max())
    if drift > 1e-3:
        raise RuntimeError(f"TorchScript output drifted from eager by {drift:.2e}")
    _sidecar(out_path, config, labels, extra)
    return out_path


def export_onnx(
    checkpoint: str | Path,
    out_path: str | Path,
    opset: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    model, config, labels, extra = load_checkpoint(checkpoint, device="cpu")
    model.eval()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    example = _example_input(config)
    dynamic_axes = {"clip": {0: "batch"}, "logits": {0: "batch"}} if dynamic_batch else None
    torch.onnx.export(
        model,
        example,
        str(out_path),
        input_names=["clip"],
        output_names=["logits"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    _sidecar(out_path, config, labels, extra)
    return out_path


@torch.no_grad()
def benchmark(
    checkpoint: str | Path | None = None,
    config=None,
    num_classes: int = 18,
    batch: int = 1,
    iterations: int = 20,
    device: str = "cpu",
) -> dict:
    """Measure real latency and throughput for one clip batch."""
    import time

    from .config import Config
    from .model import build_model

    if checkpoint is not None:
        model, config, _, _ = load_checkpoint(checkpoint, device=device)
    else:
        config = config or Config().apply_preset()
        model = build_model(config, num_classes=num_classes).to(device).eval()

    example = _example_input(config, batch).to(device)
    for _ in range(3):  # warm up allocator / kernels
        model(example)
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        model(example)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    clip_span = config.data.clip_frames * config.data.frame_stride
    clips_per_second = iterations * batch / elapsed
    return {
        "device": device,
        "batch": batch,
        "latency_ms": elapsed / iterations * 1000,
        "clips_per_second": clips_per_second,
        "realtime_factor": clips_per_second * clip_span / 30.0,
        "parameters_m": model.num_parameters() / 1e6,
        "macs_g": model.estimate_flops(
            (3, config.data.clip_frames, config.data.image_size, config.data.image_size)
        )
        / 1e9,
    }
