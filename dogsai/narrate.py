"""DogNarrator: a small language model that describes a clip in a sentence.

The second model in this package, and a different kind of thing from the first.
`DogBehaviourNet` classifies; this one *generates* — a Transformer decoder,
written from scratch on ``torch.nn`` primitives like everything else here, that
takes a conditioning vector describing what was detected and emits a natural
sentence token by token.

Conditioning is the interesting part. The vector concatenates:

* the behaviour distribution from the video model (what it was doing),
* the audio summary — vocalisation-type counts, arousal, valence, vocal fraction
  (what it said),
* clip duration.

So the narrator sees both sensors and has to compose them into one sentence. That
is genuinely more than either channel alone, and it is why this is a model rather
than an f-string.

An honest word about what it can learn
--------------------------------------
The narrator is only as expressive as its captions. The `dogbehaviour` dataset has
**five distinct caption strings** ("Dog is eating.", "Dog is yawning.", …), one per
class. Trained on that, the narrator learns a near-deterministic mapping from the
behaviour vector to one of five sentences — it will reach very high accuracy and
that accuracy means little, because the task is barely harder than the classifier
it is reading from.

That is a data limitation, not an architectural one, and it is worth being blunt
about rather than presenting a 98% caption accuracy as if it were impressive. The
architecture is sized and built for richer supervision: give it captions that
actually vary — describing intensity, sequence, or context — and the same model
learns to produce them. Until then, treat its output as a fluent restatement of the
detections, and prefer :mod:`dogsai.translate` (rule-based, but composes audio and
video with explicit provenance) when you want the reasoning to be inspectable.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

# Vocalisation kinds the conditioning vector reserves a slot for.
VOICE_KINDS: tuple[str, ...] = (
    "bark", "bark_alarm", "bark_excited", "growl", "whine", "howl", "yelp", "pant",
)


# ---------------------------------------------------------------------------
# tokenisation
# ---------------------------------------------------------------------------
class WordTokenizer:
    """Word-level tokeniser.

    Word-level rather than sub-word because the caption vocabulary here is tiny
    (tens of words). BPE would add a dependency and a failure mode to solve a
    problem this corpus does not have.
    """

    def __init__(self, vocab: list[str] | None = None):
        self.itos: list[str] = list(vocab) if vocab else list(SPECIALS)
        self.stoi: dict[str, int] = {w: i for i, w in enumerate(self.itos)}

    @staticmethod
    def split(text: str) -> list[str]:
        # Keep punctuation as its own token so the model learns to end sentences.
        return re.findall(r"[a-z0-9']+|[.,!?;]", text.lower())

    @classmethod
    def build(cls, texts: list[str], min_count: int = 1) -> "WordTokenizer":
        counts: dict[str, int] = {}
        for text in texts:
            for word in cls.split(text):
                counts[word] = counts.get(word, 0) + 1
        vocab = list(SPECIALS) + sorted(w for w, c in counts.items() if c >= min_count)
        return cls(vocab)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = [self.stoi.get(w, UNK) for w in self.split(text)]
        return [BOS] + ids + [EOS] if add_special else ids

    def decode(self, ids: list[int]) -> str:
        words = [self.itos[i] for i in ids if i not in (PAD, BOS, EOS)]
        text = " ".join(words)
        text = re.sub(r"\s+([.,!?;])", r"\1", text)   # no space before punctuation
        return text[:1].upper() + text[1:] if text else ""

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.itos))

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        return cls(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# conditioning
# ---------------------------------------------------------------------------
def condition_vector(
    behaviour_scores: np.ndarray,
    audio_summary: dict | None = None,
    duration: float = 0.0,
) -> np.ndarray:
    """Pack detections into the fixed-width vector the narrator conditions on.

    Layout: ``[behaviour scores | voice counts (normalised) | arousal, valence,
    vocal_fraction, log duration]``. Counts are squashed with ``log1p`` and scaled
    because "three barks" and "thirty barks" should not differ by 10x in a feature
    the decoder attends to once.
    """
    behaviour = np.asarray(behaviour_scores, dtype=np.float32).ravel()
    voice = np.zeros(len(VOICE_KINDS), dtype=np.float32)
    arousal = valence = fraction = 0.0
    if audio_summary:
        kinds = audio_summary.get("kinds", {}) or {}
        for i, name in enumerate(VOICE_KINDS):
            voice[i] = math.log1p(float(kinds.get(name, 0))) / 3.0
        arousal = float(audio_summary.get("arousal", 0.0) or 0.0)
        valence = float(audio_summary.get("valence", 0.0) or 0.0)
        fraction = float(audio_summary.get("vocal_fraction", 0.0) or 0.0)
    extra = np.array(
        [arousal, valence, fraction, math.log1p(max(0.0, duration)) / 4.0],
        dtype=np.float32,
    )
    return np.concatenate([behaviour, voice, extra])


def condition_dim(num_behaviours: int) -> int:
    return num_behaviours + len(VOICE_KINDS) + 4


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask, written out explicitly."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} is not divisible by heads {heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (b, t, self.heads, self.head_dim)
        q, k, v = (z.view(shape).transpose(1, 2) for z in (q, k, v))
        # Fused kernel where available; is_causal applies the mask without
        # materialising a (t, t) tensor per layer.
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.p if self.training else 0.0, is_causal=True
        )
        attended = attended.transpose(1, 2).reshape(b, t, c)
        return self.dropout(self.out(attended))


class DecoderBlock(nn.Module):
    """Pre-norm Transformer block: attention then MLP, both residual."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.attention = CausalSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


@dataclass
class NarratorConfig:
    vocab_size: int = 64
    condition_dim: int = 17
    dim: int = 128
    depth: int = 3
    heads: int = 4
    max_len: int = 48
    dropout: float = 0.1
    prefix_tokens: int = 4
    """How many tokens the conditioning vector is projected into. More than one so
    the decoder can attend to different aspects of the detection separately."""


class DogNarrator(nn.Module):
    """Conditioned Transformer decoder over caption tokens."""

    def __init__(self, config: NarratorConfig):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.dim, padding_idx=PAD)
        self.positions = nn.Parameter(
            torch.zeros(1, config.max_len + config.prefix_tokens, config.dim)
        )
        # The conditioning vector becomes a short prefix the tokens attend back to
        # — cross-attention would work too, but a prefix keeps the model a plain
        # causal decoder, which is simpler to train and to sample from.
        self.condition = nn.Sequential(
            nn.Linear(config.condition_dim, config.dim * 2),
            nn.GELU(),
            nn.Linear(config.dim * 2, config.dim * config.prefix_tokens),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            DecoderBlock(config.dim, config.heads, dropout=config.dropout)
            for _ in range(config.depth)
        )
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        # Weight tying: the input embedding and output projection describe the same
        # vocabulary, and on a corpus this small the halved parameter count is the
        # difference between learning and memorising.
        self.head.weight = self.tokens.weight

        nn.init.trunc_normal_(self.positions, std=0.02)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)
            with torch.no_grad():
                module.weight[PAD].zero_()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, condition: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """``(B, condition_dim)`` and ``(B, T)`` -> logits ``(B, T, vocab)``."""
        b, t = tokens.shape
        prefix = self.condition(condition).view(b, self.config.prefix_tokens, -1)
        embedded = self.tokens(tokens)
        x = torch.cat([prefix, embedded], dim=1)
        x = self.dropout(x + self.positions[:, : x.shape[1]])
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Drop the prefix positions: they predict nothing.
        return self.head(x[:, self.config.prefix_tokens :])

    @torch.no_grad()
    def generate(
        self,
        condition: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int = 0,
    ) -> list[int]:
        """Sample one caption. ``temperature=0`` is greedy (and deterministic)."""
        self.eval()
        device = next(self.parameters()).device
        if condition.ndim == 1:
            condition = condition.unsqueeze(0)
        condition = condition.to(device)
        tokens = torch.tensor([[BOS]], dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            window = tokens[:, -self.config.max_len :]
            logits = self(condition, window)[:, -1, :]
            logits[:, PAD] = -float("inf")   # never emit padding
            if temperature <= 0:
                nxt = int(logits.argmax(dim=-1))
            else:
                logits = logits / temperature
                if top_k:
                    kth = logits.topk(min(top_k, logits.shape[-1])).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, -float("inf"))
                nxt = int(torch.multinomial(logits.softmax(dim=-1), 1))
            if nxt == EOS:
                break
            tokens = torch.cat(
                [tokens, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1
            )
        return tokens[0, 1:].tolist()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@dataclass
class NarrationExample:
    condition: np.ndarray
    caption: str


class NarrationDataset(torch.utils.data.Dataset):
    """Pads captions to a common length and returns teacher-forcing pairs."""

    def __init__(self, examples: list[NarrationExample], tokenizer: WordTokenizer, max_len: int = 48):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len
        if not examples:
            raise ValueError("no narration examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        ids = self.tokenizer.encode(example.caption)[: self.max_len + 1]
        inputs = ids[:-1]
        targets = ids[1:]
        pad = self.max_len - len(inputs)
        return {
            "condition": torch.from_numpy(example.condition.astype(np.float32)),
            "tokens": torch.tensor(inputs + [PAD] * pad, dtype=torch.long),
            "targets": torch.tensor(targets + [PAD] * pad, dtype=torch.long),
        }


def build_examples(
    annotations,
    behaviours: list[str],
    with_audio: bool = True,
    verbose: bool = False,
) -> list[NarrationExample]:
    """Build (condition, caption) pairs from annotations.

    Conditioning uses the *ground-truth* behaviour vector during training rather
    than the video model's prediction. That is a deliberate choice: it decouples
    the narrator from whichever classifier checkpoint happens to exist, so the two
    models can be trained and improved independently. The cost is a train/inference
    mismatch — at inference the behaviour vector is a soft prediction, not a clean
    one-hot — which is mitigated by label smoothing on the condition below.
    """
    from .audio import read_audio

    index = {name: i for i, name in enumerate(behaviours)}
    examples: list[NarrationExample] = []
    for n, annotation in enumerate(annotations):
        caption = (annotation.meta or {}).get("caption")
        if not caption:
            # Fall back to a caption derived from the label, so a dataset without
            # free-text captions still trains something coherent.
            names = " and ".join(a.replace("_", " ") for a in annotation.labels)
            caption = f"Dog is {names}." if names else "Nothing notable."
        scores = np.zeros(len(behaviours), dtype=np.float32)
        for label in annotation.labels:
            if label in index:
                scores[index[label]] = 1.0
        if scores.sum() > 0:
            # Smooth the one-hot toward the soft distributions seen at inference.
            scores = scores * 0.85 + 0.15 / len(behaviours)

        summary = None
        if with_audio:
            try:
                summary = read_audio(annotation.video).to_dict()
            except Exception:
                summary = None
        duration = (annotation.end or 0.0) - annotation.start
        examples.append(
            NarrationExample(
                condition=condition_vector(scores, summary, duration),
                caption=caption,
            )
        )
        if verbose and (n + 1) % 100 == 0:
            print(f"  featurised {n + 1}/{len(annotations)}", flush=True)
    return examples


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
@dataclass
class NarratorTrainConfig:
    epochs: int = 60
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_epochs: float = 3.0
    label_smoothing: float = 0.05
    grad_clip: float = 1.0
    seed: int = 0
    out_dir: str = "runs/narrator"


@dataclass
class NarratorResult:
    epochs: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_perplexity: float = 0.0
    exact_match: float = 0.0
    history: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"loss {self.val_loss:.4f}  perplexity {self.val_perplexity:.2f}  "
            f"exact-match captions {self.exact_match:.1%}"
        )


def train_narrator(
    train_examples: list[NarrationExample],
    val_examples: list[NarrationExample],
    tokenizer: WordTokenizer,
    model_config: NarratorConfig | None = None,
    train_config: NarratorTrainConfig | None = None,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> tuple[DogNarrator, NarratorResult]:
    """Train the narrator with teacher forcing."""
    train_config = train_config or NarratorTrainConfig()
    model_config = model_config or NarratorConfig(
        vocab_size=len(tokenizer),
        condition_dim=len(train_examples[0].condition),
    )
    torch.manual_seed(train_config.seed)
    device = torch.device(device)

    model = DogNarrator(model_config).to(device)
    train_set = NarrationDataset(train_examples, tokenizer, model_config.max_len)
    val_set = NarrationDataset(val_examples, tokenizer, model_config.max_len) if val_examples else None
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=train_config.batch_size, shuffle=True, drop_last=False
    )
    val_loader = (
        torch.utils.data.DataLoader(val_set, batch_size=train_config.batch_size)
        if val_set
        else None
    )

    decay = [p for n, p in model.named_parameters() if p.ndim > 1 and "positions" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1 or "positions" in n]
    optimiser = torch.optim.AdamW(
        [{"params": decay, "weight_decay": train_config.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=train_config.lr,
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * train_config.epochs
    warmup = int(steps_per_epoch * train_config.warmup_epochs)
    result = NarratorResult()
    step = 0
    best = float("inf")

    for epoch in range(train_config.epochs):
        model.train()
        running, seen = 0.0, 0
        for batch in train_loader:
            lr = (
                train_config.lr * (step + 1) / max(1, warmup)
                if step < warmup
                else train_config.lr
                * 0.5
                * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
            )
            for group in optimiser.param_groups:
                group["lr"] = lr
            condition = batch["condition"].to(device)
            tokens = batch["tokens"].to(device)
            targets = batch["targets"].to(device)
            logits = model(condition, tokens)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=PAD,
                label_smoothing=train_config.label_smoothing,
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            if train_config.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimiser.step()
            running += float(loss.detach()) * tokens.shape[0]
            seen += tokens.shape[0]
            step += 1

        record = {"epoch": epoch + 1, "train_loss": running / max(1, seen), "lr": lr}
        if val_loader is not None:
            model.eval()
            total, count = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    condition = batch["condition"].to(device)
                    tokens = batch["tokens"].to(device)
                    targets = batch["targets"].to(device)
                    logits = model(condition, tokens)
                    # Unsmoothed for reporting: perplexity should be comparable.
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        targets.reshape(-1),
                        ignore_index=PAD,
                    )
                    total += float(loss.detach()) * tokens.shape[0]
                    count += tokens.shape[0]
            record["val_loss"] = total / max(1, count)
            record["val_perplexity"] = math.exp(min(20.0, record["val_loss"]))
            if record["val_loss"] < best:
                best = record["val_loss"]
        result.history.append(record)
        if verbose and (epoch + 1) % max(1, train_config.epochs // 10) == 0:
            extra = (
                f"  val_loss={record['val_loss']:.4f} ppl={record['val_perplexity']:.2f}"
                if "val_loss" in record
                else ""
            )
            print(f"  epoch {epoch + 1}/{train_config.epochs} "
                  f"loss={record['train_loss']:.4f}{extra}", flush=True)

    result.epochs = train_config.epochs
    result.train_loss = result.history[-1]["train_loss"]
    result.val_loss = result.history[-1].get("val_loss", 0.0)
    result.val_perplexity = result.history[-1].get("val_perplexity", 0.0)

    # Exact-match generation accuracy: the honest end-to-end number.
    if val_examples:
        matches = 0
        for example in val_examples:
            condition = torch.from_numpy(example.condition.astype(np.float32))
            text = tokenizer.decode(model.generate(condition))
            if text.strip().lower() == example.caption.strip().lower().rstrip("."):
                matches += 1
            elif text.strip().lower().rstrip(".") == example.caption.strip().lower().rstrip("."):
                matches += 1
        result.exact_match = matches / len(val_examples)
    return model, result


def save_narrator(
    path: str | Path,
    model: DogNarrator,
    tokenizer: WordTokenizer,
    behaviours: list[str],
    result: NarratorResult | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(model.config),
            "vocab": tokenizer.itos,
            "behaviours": behaviours,
            "metrics": asdict(result) if result else {},
        },
        path,
    )
    return path


def load_narrator(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[DogNarrator, WordTokenizer, list[str], dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = NarratorConfig(**payload["config"])
    model = DogNarrator(config)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    tokenizer = WordTokenizer(payload["vocab"])
    return model, tokenizer, payload.get("behaviours", []), payload.get("metrics", {})


def narrate(
    model: DogNarrator,
    tokenizer: WordTokenizer,
    behaviour_scores: np.ndarray,
    audio_summary: dict | None = None,
    duration: float = 0.0,
    temperature: float = 0.0,
) -> str:
    """Generate one sentence describing a clip."""
    condition = torch.from_numpy(
        condition_vector(behaviour_scores, audio_summary, duration).astype(np.float32)
    )
    return tokenizer.decode(model.generate(condition, temperature=temperature))
