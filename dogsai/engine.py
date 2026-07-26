"""Losses, the training loop and checkpointing."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .dataset import ClipDataset, build_sampler, collate
from .labels import LabelSpace
from .metrics import EvalResult, evaluate
from .model import DogBehaviourNet, ModelEMA, build_model
from .transforms import mixup_batch

CHECKPOINT_VERSION = 1


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
class SoftTargetCrossEntropy(nn.Module):
    """Cross-entropy against soft targets, with label smoothing.

    Written to take a full probability vector rather than a class index so that
    mixup and smoothing compose: mixup already produces soft targets, and the
    usual ``label_smoothing=`` argument of ``F.cross_entropy`` cannot be applied on
    top of them.
    """

    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = smoothing

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        n_classes = logits.shape[1]
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + self.smoothing / n_classes
        loss = -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        if weight is not None:
            loss = loss * weight
        return loss.mean()


class FocalBCELoss(nn.Module):
    """Binary cross-entropy with a focal term and optional positive weighting.

    Multi-label behaviour data is dominated by negatives — with 18 classes and ~2
    active per clip, 89% of the targets are zero.  Plain BCE lets that majority
    set the gradient; the focal factor ``(1-p_t)^gamma`` down-weights the easy
    negatives so rare behaviours keep contributing signal.

    ``pos_weight`` is derived from class frequency by
    :meth:`Trainer._build_criterion` and clamped, because uncapped inverse
    frequency on a class with three examples produces a weight of ~300 and a model
    that predicts it constantly.
    """

    def __init__(
        self,
        gamma: float = 1.5,
        pos_weight: torch.Tensor | None = None,
        smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing
        self.register_buffer(
            "pos_weight", pos_weight if pos_weight is not None else torch.tensor([])
        )

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        targets = targets.float()
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        pos_weight = self.pos_weight if self.pos_weight.numel() else None
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="none"
        )
        if self.gamma > 0:
            probability = torch.sigmoid(logits)
            p_t = probability * targets + (1 - probability) * (1 - targets)
            loss = loss * (1 - p_t).clamp_min(1e-6).pow(self.gamma)
        if weight is not None:
            loss = loss * weight[:, None]
        return loss.mean()


# ---------------------------------------------------------------------------
# schedule
# ---------------------------------------------------------------------------
def cosine_lr(
    step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float
) -> float:
    """Linear warmup then cosine decay, evaluated per optimiser step."""
    if total_steps <= 0:
        return base_lr
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Split parameters so norms, biases and the attention query skip weight decay.

    Decaying a BatchNorm gain or a bias is a small but real accuracy loss, and
    decaying the learned pooling query actively fights the pooling mechanism.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "query" in name or "gain" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------
def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: Config,
    labels: LabelSpace,
    epoch: int = 0,
    metrics: dict | None = None,
    thresholds: np.ndarray | None = None,
    optimiser: torch.optim.Optimizer | None = None,
    ema: ModelEMA | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "model": model.state_dict(),
        "config": config.to_dict(),
        "behaviours": labels.to_list(),
        "epoch": epoch,
        "metrics": metrics or {},
        "thresholds": None if thresholds is None else [float(t) for t in thresholds],
    }
    if optimiser is not None:
        payload["optimiser"] = optimiser.state_dict()
    if ema is not None:
        payload["ema"] = ema.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
    prefer_ema: bool = True,
) -> tuple[DogBehaviourNet, Config, LabelSpace, dict]:
    """Rebuild a ready-to-use model from a checkpoint.

    A checkpoint carries its own config and label space, so inference never needs
    the training config file — passing a mismatched one is a classic source of
    silently wrong predictions.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    config = Config.from_dict(payload["config"])
    labels = LabelSpace.from_names(payload["behaviours"])
    model = build_model(config, num_classes=len(labels))
    state = payload["ema"] if (prefer_ema and payload.get("ema")) else payload["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match the model built from its config "
            f"(missing={list(missing)[:4]}, unexpected={list(unexpected)[:4]})"
        )
    model.to(device).eval()
    extra = {
        "epoch": payload.get("epoch", 0),
        "metrics": payload.get("metrics", {}),
        "thresholds": payload.get("thresholds"),
        "used_ema": state is payload.get("ema"),
    }
    return model, config, labels, extra


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------
@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0
    best_metric: float = -float("inf")
    best_epoch: int = -1
    history: list[dict] = field(default_factory=list)


class Trainer:
    """Owns the training loop, evaluation, checkpointing and early stopping."""

    def __init__(
        self,
        config: Config,
        train_dataset: ClipDataset,
        val_dataset: ClipDataset | None = None,
        labels: LabelSpace | None = None,
        device: str | torch.device | None = None,
        verbose: bool = True,
    ):
        self.config = config
        self.labels = labels or LabelSpace.from_names(config.behaviours)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.verbose = verbose
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.out_dir = Path(config.train.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(config.train.seed)
        np.random.seed(config.train.seed)

        self.model = build_model(config, num_classes=len(self.labels))
        positives = train_dataset.positives_per_class() / max(1, len(train_dataset))
        if config.task == "multilabel":
            self.model.set_prior_bias(positives)
        self.model.to(self.device)
        if config.train.channels_last and self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last_3d)
        if config.train.compile:
            self.model = torch.compile(self.model)  # type: ignore[assignment]

        self.criterion = self._build_criterion(positives).to(self.device)
        self.optimiser = torch.optim.AdamW(
            param_groups(self.model, config.train.weight_decay),
            lr=config.train.lr,
            betas=(0.9, 0.999),
        )
        self.ema = (
            ModelEMA(self.model, decay=config.train.ema_decay).to(self.device)
            if config.train.ema_decay > 0
            else None
        )
        self.use_amp = config.train.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        self.train_loader = self._make_loader(train_dataset, training=True)
        self.val_loader = (
            self._make_loader(val_dataset, training=False) if val_dataset else None
        )
        self.state = TrainState()
        self.thresholds: np.ndarray | None = None

    # -- setup helpers ---------------------------------------------------
    def _build_criterion(self, positive_rate: np.ndarray) -> nn.Module:
        cfg = self.config.train
        if self.config.task == "multiclass":
            return SoftTargetCrossEntropy(smoothing=cfg.label_smoothing)
        rate = np.clip(positive_rate, 1e-3, 1.0)
        # sqrt of inverse frequency, capped: a gentler correction than 1/f, which
        # over-corrects hard on very rare classes.
        weights = np.sqrt((1.0 - rate) / rate)
        weights = np.clip(weights, 1.0, 10.0)
        return FocalBCELoss(
            gamma=cfg.focal_gamma,
            pos_weight=torch.tensor(weights, dtype=torch.float32),
            smoothing=cfg.label_smoothing * 0.5,
        )

    def _make_loader(self, dataset: ClipDataset, training: bool) -> DataLoader:
        cfg = self.config
        sampler = build_sampler(dataset, cfg) if training else None
        workers = cfg.data.num_workers
        kwargs = {}
        if workers > 0:
            kwargs = {
                "prefetch_factor": cfg.data.prefetch_factor,
                "persistent_workers": True,
            }
        return DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            sampler=sampler,
            shuffle=(training and sampler is None),
            num_workers=workers,
            pin_memory=cfg.data.pin_memory and self.device.type == "cuda",
            drop_last=training and len(dataset) > cfg.train.batch_size,
            collate_fn=collate,
            **kwargs,
        )

    # -- loop ------------------------------------------------------------
    @property
    def steps_per_epoch(self) -> int:
        return max(1, len(self.train_loader) // max(1, self.config.train.accum_steps))

    def _set_lr(self) -> float:
        cfg = self.config.train
        total = self.steps_per_epoch * cfg.epochs
        warmup = int(self.steps_per_epoch * cfg.warmup_epochs)
        lr = cosine_lr(self.state.step, total, warmup, cfg.lr, cfg.min_lr)
        for group in self.optimiser.param_groups:
            group["lr"] = lr
        return lr

    def train_one_epoch(self) -> dict:
        cfg = self.config.train
        self.model.train()
        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.state.epoch)

        running, seen, lr = 0.0, 0, cfg.lr
        started = time.time()
        self.optimiser.zero_grad(set_to_none=True)

        for i, batch in enumerate(self.train_loader):
            clips = batch["clip"].to(self.device, non_blocking=True)
            targets = batch["target"].to(self.device, non_blocking=True)
            weights = batch["weight"].to(self.device, non_blocking=True)
            if cfg.channels_last and self.device.type == "cuda":
                clips = clips.to(memory_format=torch.channels_last_3d)
            if cfg.mixup > 0:
                clips, targets = mixup_batch(clips, targets, cfg.mixup)

            with torch.autocast("cuda", enabled=self.use_amp):
                logits = self.model(clips)
                loss = self.criterion(logits, targets, weights)
            self.scaler.scale(loss / cfg.accum_steps).backward()

            if (i + 1) % cfg.accum_steps == 0:
                lr = self._set_lr()
                if cfg.grad_clip > 0:
                    self.scaler.unscale_(self.optimiser)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.scaler.step(self.optimiser)
                self.scaler.update()
                self.optimiser.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update(self.model)
                self.state.step += 1

            running += float(loss.detach()) * clips.shape[0]
            seen += clips.shape[0]
            if self.verbose and cfg.log_every and (i + 1) % cfg.log_every == 0:
                rate = seen / max(1e-9, time.time() - started)
                print(
                    f"  epoch {self.state.epoch + 1} "
                    f"[{i + 1}/{len(self.train_loader)}] "
                    f"loss={running / max(1, seen):.4f} lr={lr:.2e} "
                    f"{rate:.1f} clips/s",
                    flush=True,
                )

        return {
            "loss": running / max(1, seen),
            "lr": lr,
            "clips_per_second": seen / max(1e-9, time.time() - started),
        }

    @torch.no_grad()
    def predict_loader(self, loader: DataLoader, use_ema: bool = True) -> tuple[np.ndarray, np.ndarray]:
        model = self.ema.module if (use_ema and self.ema is not None) else self.model
        model.eval()
        all_logits, all_targets = [], []
        for batch in loader:
            clips = batch["clip"].to(self.device, non_blocking=True)
            if self.config.train.channels_last and self.device.type == "cuda":
                clips = clips.to(memory_format=torch.channels_last_3d)
            with torch.autocast("cuda", enabled=self.use_amp):
                logits = model(clips)
            all_logits.append(logits.float().cpu().numpy())
            all_targets.append(batch["target"].numpy())
        model.train()
        if not all_logits:
            raise RuntimeError("evaluation loader produced no batches")
        return np.concatenate(all_logits), np.concatenate(all_targets)

    def evaluate(self, use_ema: bool = True) -> EvalResult | None:
        if self.val_loader is None:
            return None
        logits, targets = self.predict_loader(self.val_loader, use_ema=use_ema)
        result = evaluate(logits, targets, self.labels.to_list(), task=self.config.task)
        if result.thresholds is not None:
            self.thresholds = result.thresholds
        return result

    def fit(self) -> TrainState:
        cfg = self.config.train
        if self.verbose:
            print(self.model_summary(), flush=True)
            print(
                f"train clips: {len(self.train_dataset)}"
                + (f"  val clips: {len(self.val_dataset)}" if self.val_dataset else "")
                + f"  device: {self.device}  amp: {self.use_amp}",
                flush=True,
            )
        self.config.save(self.out_dir / "config.json")
        since_improved = 0

        for epoch in range(cfg.epochs):
            self.state.epoch = epoch
            train_stats = self.train_one_epoch()
            result = self.evaluate()
            record = {"epoch": epoch + 1, **train_stats}
            if result is not None:
                record.update({"val_" + k: v for k, v in result.to_dict().items() if isinstance(v, float)})
            self.state.history.append(record)

            metric = result.primary if result is not None else -train_stats["loss"]
            improved = metric > self.state.best_metric
            if improved and not (isinstance(metric, float) and math.isnan(metric)):
                self.state.best_metric = metric
                self.state.best_epoch = epoch
                since_improved = 0
                self.save("best.pt", result)
            else:
                since_improved += 1
            self.save("last.pt", result)

            if self.verbose:
                line = (
                    f"epoch {epoch + 1}/{cfg.epochs}  loss={train_stats['loss']:.4f}  "
                    f"{train_stats['clips_per_second']:.1f} clips/s"
                )
                if result is not None:
                    line += f"  val {result.summary()}"
                if improved:
                    line += "  *best*"
                print(line, flush=True)

            (self.out_dir / "history.json").write_text(
                json.dumps(self.state.history, indent=2) + "\n"
            )
            if cfg.early_stop_patience and since_improved >= cfg.early_stop_patience:
                if self.verbose:
                    print(
                        f"early stop: no improvement for {since_improved} epochs "
                        f"(best {self.state.best_metric:.4f} @ epoch {self.state.best_epoch + 1})",
                        flush=True,
                    )
                break

        if self.verbose and self.val_loader is not None:
            name = "map" if self.config.task == "multilabel" else "balanced_acc"
            print(
                f"\nbest {name}: {self.state.best_metric:.4f} "
                f"@ epoch {self.state.best_epoch + 1}"
            )
        return self.state

    def save(self, name: str, result: EvalResult | None = None) -> Path:
        return save_checkpoint(
            self.out_dir / name,
            self.model,
            self.config,
            self.labels,
            epoch=self.state.epoch + 1,
            metrics=result.to_dict() if result else {},
            thresholds=self.thresholds,
            ema=self.ema,
        )

    def model_summary(self) -> str:
        shape = (3, self.config.data.clip_frames, self.config.data.image_size, self.config.data.image_size)
        model = self.model
        return model.summary(shape) if isinstance(model, DogBehaviourNet) else repr(model)
