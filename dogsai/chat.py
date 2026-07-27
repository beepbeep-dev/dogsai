"""DogChat: ask questions about a clip and get answers, plus a dialogue dataset.

The third model here, and the one you can talk to. Given a clip's detections it
answers free-text questions — "how are you feeling?", "are you hurt?", "what do
you want?", "how should I approach you?" — in two voices: the dog's, and a
handler's. It is a conditional language model, written on the same from-scratch
Transformer decoder as :mod:`dogsai.narrate`, trained on a dialogue corpus this
module generates.

Where the training data comes from
----------------------------------
There is no public corpus of "dog state + human question -> answer", so this module
builds one. :func:`generate_dialogues` enumerates the cross product of

* **dog states** — behaviour, vocalisation type, arousal and valence bands, drawn
  from the ranges the detectors actually produce, and
* **intents** — the dozen or so things people actually ask about their dog,

and composes a question and a two-part answer for each combination, with varied
phrasing on both sides. Tens of thousands of pairs, generated deterministically so
the corpus is reproducible.

Be clear about what that means. The knowledge in the answers is **mine, encoded as
templates** — it comes from conventional dog-behaviour guidance, not from data. The
model's job is to learn the *mapping*: which state and which question select which
answer, and how to phrase it fluently. So:

* it can generalise over phrasings of a question it was trained on, and interpolate
  between states, which a lookup table cannot;
* it cannot know anything the templates did not encode, and it will answer
  confidently anyway when asked something outside them. :func:`DogChat.answer`
  reports an out-of-domain score for exactly this reason;
* it is not a source of veterinary or behavioural authority. It is a fluent
  interface onto the same conservative guidance as :mod:`dogsai.advise`.

The honest framing: a *distilled conversational interface* over the detections and
a fixed body of advice — genuinely useful for interrogating a clip, and not an
oracle about your dog.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .narrate import (
    BOS,
    EOS,
    PAD,
    VOICE_KINDS,
    DecoderBlock,
    WordTokenizer,
)

SEP = 4  # separates the question from the answer
CHAT_SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>", "<sep>"]

# ---------------------------------------------------------------------------
# the intents people actually ask about
# ---------------------------------------------------------------------------
INTENTS: tuple[str, ...] = (
    "feeling", "want", "hurt", "happy", "why_noise", "approach",
    "tired", "worry", "doing", "hungry", "play", "calm_down",
)

_QUESTIONS: dict[str, tuple[str, ...]] = {
    "feeling": ("how are you feeling", "how do you feel", "what is your mood",
                "how is my dog feeling", "are you okay"),
    "want": ("what do you want", "what are you asking for", "what do you need",
             "what is it you want"),
    "hurt": ("are you hurt", "are you in pain", "does something hurt",
             "is my dog injured"),
    "happy": ("are you happy", "are you enjoying this", "is my dog happy",
              "are you having a good time"),
    "why_noise": ("why are you barking", "why the noise", "what is that sound about",
                  "why are you making noise", "what are you saying"),
    "approach": ("how should i approach you", "can i come closer", "how do i talk to you",
                 "should i pet you now", "how should i handle this"),
    "tired": ("are you tired", "do you need a rest", "are you sleepy"),
    "worry": ("should i be worried", "is anything wrong", "do i need to do something",
              "is this a problem"),
    "doing": ("what are you doing", "what is happening", "what is going on"),
    "hungry": ("are you hungry", "do you want food", "is this about food"),
    "play": ("do you want to play", "shall we play", "do you want the toy"),
    "calm_down": ("how do i calm you down", "how can i settle you",
                  "how do i help you relax"),
}

# Dog-voice answers, keyed by intent and the state that matters for it.
_DOG_VOICE: dict[str, dict[str, tuple[str, ...]]] = {
    "feeling": {
        "positive_high": ("brilliant, honestly, i could do this all day",
                          "great, i am full of it right now"),
        "positive_low": ("comfortable, nothing needs doing",
                         "settled and fine, thank you"),
        "negative_high": ("wound up and not happy about it",
                          "not good, and i am telling you loudly"),
        "negative_low": ("a bit flat, i would rather be elsewhere",
                         "not great, i have gone quiet about it"),
        "neutral": ("alright, nothing much either way", "fine, just getting on with it"),
    },
    "want": {
        "whine": ("something i cannot reach myself, please help",
                  "your help with something"),
        "play": ("more of this, throw it again", "to keep playing"),
        "eating_drinking": ("to be left to my food", "nothing, i am eating"),
        "growl": ("space, please, and for that to stop",
                  "you to move back a little"),
        "default": ("nothing urgent", "not much, i am content"),
    },
    "hurt": {
        "yelp": ("that hurt, something is wrong",
                 "yes, something just hurt me"),
        "growl": ("nothing hurts but something is bothering me",
                  "not hurt, but do not push me"),
        "default": ("nothing hurts as far as i can tell",
                    "no, i feel fine"),
    },
    "happy": {
        "positive_high": ("yes, very", "yes, this is the good part of the day"),
        "positive_low": ("content, yes", "quietly happy, yes"),
        "negative_high": ("no, i am not enjoying this",
                          "no, this is too much for me"),
        "negative_low": ("not really, no", "no, i have gone quiet"),
        "neutral": ("neither really", "i am alright"),
    },
    "why_noise": {
        "bark_alarm": ("something is out there and i do not like it",
                       "i noticed something and i want it gone"),
        "bark_excited": ("because this is exciting and you are here",
                         "because i am pleased about something"),
        "growl": ("i am warning you, that is close enough",
                  "because i want more space"),
        "whine": ("because i need something", "because i am asking you for help"),
        "howl": ("because i do not want to be the only one here",
                 "to see if anyone answers"),
        "yelp": ("because that hurt", "because something startled me"),
        "default": ("i am not making much noise", "i have been quiet"),
    },
    "tired": {
        "yawning": ("possibly, or i am just a bit unsettled",
                    "maybe, it has been a long one"),
        "positive_low": ("yes, i could rest", "a bit, yes"),
        "default": ("not especially", "no, i have energy"),
    },
    "doing": {
        "default": ("what it looks like i am doing", "getting on with things"),
    },
    "hungry": {
        "eating_drinking": ("i am eating right now", "yes, and i am dealing with it"),
        "whine": ("possibly, i am asking for something",
                  "it might be about food"),
        "default": ("not that i am aware of", "no, this is not about food"),
    },
    "play": {
        "playing": ("yes, obviously, throw it", "already playing, keep going"),
        "positive_high": ("yes, please", "yes, i am up for it"),
        "negative_high": ("not right now", "no, i am not in the mood"),
        "default": ("maybe later", "not particularly"),
    },
}

# Handler-voice guidance, keyed the same way. Conservative and non-aversive by
# construction, matching dogsai.advise.
_HANDLER_VOICE: dict[str, dict[str, tuple[str, ...]]] = {
    "approach": {
        "growl": ("do not approach. give it space, do not lean over it or reach for "
                  "it, and work out what it wanted distance from.",),
        "alert_freeze": ("do not approach while it is fixed on something. increase "
                         "the distance calmly and let it disengage first.",),
        "negative_high": ("keep your distance for now, keep your voice low, and let "
                          "it come to you rather than reaching in.",),
        "positive_high": ("approach normally but stay side on, and let it choose the "
                          "contact. it is excited, so keep your movements slow.",),
        "eating_drinking": ("leave it to eat. approaching a dog at its bowl is how "
                            "resource guarding gets taught.",),
        "eliminating": ("give it a moment of privacy and approach afterwards.",),
        "default": ("approach calmly from the side, crouch rather than lean, and let "
                    "it come the last step to you.",),
    },
    "worry": {
        "yelp": ("yes, worth checking. a yelp suggests sudden pain. look it over "
                 "gently and see a vet if it recurs, limps or does not settle.",),
        "growl": ("take it seriously but do not punish it. the growl is a warning, "
                  "and suppressing it removes the warning rather than the reason.",),
        "alert_freeze": ("yes. freezing with a hard stare often comes before "
                         "escalation. increase distance now.",),
        "negative_high": ("worth attention. reduce whatever is driving it and give "
                          "the dog somewhere quiet it can choose.",),
        "default": ("nothing here looks alarming. keep an eye on it and trust "
                    "changes from its own baseline over any single moment.",),
    },
    "calm_down": {
        "negative_high": ("lower the stimulation: fewer people, less noise, more "
                          "distance from whatever set it off. do not crowd it or "
                          "hold it still. offer a quiet space it can choose.",),
        "positive_high": ("let the excitement burn down rather than escalating it. "
                          "stop the game, stand still, wait for a pause and reward "
                          "the pause.",),
        "default": ("it is not especially wound up. quiet company is enough.",),
    },
    "why_noise": {
        "bark_alarm": ("identify the trigger and reduce the exposure with distance "
                       "or a barrier rather than out-shouting it.",),
        "whine": ("check the obvious needs first: water, the toilet, something out "
                  "of reach, too hot or too cold.",),
        "default": ("work out what it is directed at before responding to the noise.",),
    },
    "hurt": {
        "yelp": ("check it over gently, especially feet and joints, and get it seen "
                 "if it recurs or it is limping.",),
        "default": ("nothing in this clip suggests pain, but you know its normal "
                    "better than any model does.",),
    },
    "want": {
        "whine": ("run through the basics: water, toilet, something stuck, "
                  "temperature.",),
        "growl": ("give it space first, then work out what it was guarding or "
                  "avoiding.",),
        "default": ("nothing needed right now.",),
    },
}


# ---------------------------------------------------------------------------
# state description
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DogState:
    """The conditioning state: what the detectors saw and heard."""

    behaviour: str
    voice: str          # a vocalisation kind, or "" for silence
    arousal: float
    valence: float

    def keys(self) -> list[str]:
        """Candidate lookup keys, most specific first.

        Ordering matters: a growl outranks the posture it happened during, because
        the growl is the informative signal. This is the same precedence
        :mod:`dogsai.translate` applies.
        """
        out: list[str] = []
        if self.voice:
            out.append(self.voice)
        out.append(self.behaviour)
        band = "high" if self.arousal >= 0.55 else "low"
        if self.valence >= 0.2:
            out.append(f"positive_{band}")
        elif self.valence <= -0.15:
            out.append(f"negative_{band}")
        else:
            out.append("neutral")
        out.append("default")
        return out

    def vector(self, behaviours: Sequence[str]) -> np.ndarray:
        """Fixed-width conditioning vector, same layout family as the narrator."""
        scores = np.zeros(len(behaviours), dtype=np.float32)
        if self.behaviour in behaviours:
            scores[list(behaviours).index(self.behaviour)] = 0.85
        scores += 0.15 / max(1, len(behaviours))
        voice = np.zeros(len(VOICE_KINDS), dtype=np.float32)
        if self.voice in VOICE_KINDS:
            voice[VOICE_KINDS.index(self.voice)] = 1.0
        return np.concatenate([
            scores, voice,
            np.array([self.arousal, self.valence], dtype=np.float32),
        ])


def state_dim(num_behaviours: int) -> int:
    return num_behaviours + len(VOICE_KINDS) + 2


def _resolve(table: dict[str, tuple[str, ...]], state: DogState) -> tuple[str, ...] | None:
    for key in state.keys():
        if key in table:
            return table[key]
    return None


def _pick(options: Sequence[str], rng: np.random.Generator) -> str:
    return str(options[int(rng.integers(0, len(options)))])


def compose_answer(state: DogState, intent: str, seed: int) -> str:
    """Compose the two-voice answer for one (state, intent) pair."""
    rng = np.random.default_rng(seed)
    parts: list[str] = []

    dog_table = _DOG_VOICE.get(intent)
    if dog_table:
        options = _resolve(dog_table, state)
        if options:
            lead = _pick(("i would say", "if i could talk", "in words"), rng)
            parts.append(f"{lead} : {_pick(options, rng)} .")

    handler_table = _HANDLER_VOICE.get(intent)
    if handler_table:
        options = _resolve(handler_table, state)
        if options:
            lead = _pick(("what to do", "how to handle it", "your move"), rng)
            parts.append(f"{lead} : {_pick(options, rng)}")

    if not parts:
        parts.append("i would say : nothing much to report .")
    return " ".join(parts)


@dataclass
class DialogueExample:
    question: str
    answer: str
    state: DogState
    intent: str


def generate_dialogues(
    behaviours: Sequence[str],
    voices: Sequence[str] = ("",) + VOICE_KINDS,
    arousals: Sequence[float] = (0.15, 0.4, 0.7, 0.9),
    valences: Sequence[float] = (-0.7, -0.3, 0.0, 0.4, 0.8),
    repeats: int = 2,
    verbose: bool = True,
) -> list[DialogueExample]:
    """Enumerate the state x intent grid and compose a dialogue for each cell.

    ``repeats`` re-samples the phrasing of the same cell, which is where the
    variation the model needs comes from: identical meaning, different words, so it
    learns the mapping rather than a string.
    """
    out: list[DialogueExample] = []
    for behaviour in behaviours:
        for voice in voices:
            for arousal in arousals:
                for valence in valences:
                    state = DogState(behaviour, voice, arousal, valence)
                    for intent in INTENTS:
                        for r in range(repeats):
                            digest = hashlib.sha256(
                                f"{behaviour}|{voice}|{arousal}|{valence}|{intent}|{r}".encode()
                            ).digest()
                            seed = int.from_bytes(digest[:8], "big")
                            rng = np.random.default_rng(seed)
                            question = _pick(_QUESTIONS[intent], rng)
                            answer = compose_answer(state, intent, seed)
                            out.append(DialogueExample(question, answer, state, intent))
    if verbose:
        print(f"  generated {len(out)} dialogue pairs "
              f"({len(set(d.answer for d in out))} distinct answers)")
    return out


def dialogue_stats(examples: Sequence[DialogueExample]) -> dict:
    answers = [e.answer for e in examples]
    questions = [e.question for e in examples]
    words = {w for a in answers for w in a.split()} | {w for q in questions for w in q.split()}
    return {
        "pairs": len(examples),
        "distinct_answers": len(set(answers)),
        "distinct_questions": len(set(questions)),
        "vocabulary": len(words),
        "intents": len({e.intent for e in examples}),
        "mean_answer_words": float(np.mean([len(a.split()) for a in answers])),
    }


def save_dialogues(path: str | Path, examples: Sequence[DialogueExample]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for item in examples:
            handle.write(json.dumps({
                "question": item.question,
                "answer": item.answer,
                "intent": item.intent,
                "state": {
                    "behaviour": item.state.behaviour,
                    "voice": item.state.voice,
                    "arousal": item.state.arousal,
                    "valence": item.state.valence,
                },
            }) + "\n")
    return path


def load_dialogues(path: str | Path) -> list[DialogueExample]:
    out: list[DialogueExample] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        s = record["state"]
        out.append(DialogueExample(
            question=record["question"],
            answer=record["answer"],
            intent=record.get("intent", ""),
            state=DogState(s["behaviour"], s["voice"], float(s["arousal"]), float(s["valence"])),
        ))
    return out


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
@dataclass
class ChatConfig:
    vocab_size: int = 256
    state_dim: int = 15
    dim: int = 192
    depth: int = 4
    heads: int = 6
    max_len: int = 96
    dropout: float = 0.1
    prefix_tokens: int = 4


class DogChat(nn.Module):
    """Causal decoder over ``[state prefix] question <sep> answer``.

    One sequence rather than an encoder-decoder: the question and answer live in the
    same stream separated by ``<sep>``, so a plain causal decoder handles both, and
    the state arrives as a learned prefix. Fewer moving parts than cross-attention
    and it trains stably on a corpus this size.
    """

    def __init__(self, config: ChatConfig):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(config.vocab_size, config.dim, padding_idx=PAD)
        self.positions = nn.Parameter(
            torch.zeros(1, config.max_len + config.prefix_tokens, config.dim)
        )
        self.state = nn.Sequential(
            nn.Linear(config.state_dim, config.dim * 2), nn.GELU(),
            nn.Linear(config.dim * 2, config.dim * config.prefix_tokens),
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            DecoderBlock(config.dim, config.heads, dropout=config.dropout)
            for _ in range(config.depth)
        )
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
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

    def forward(self, state: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        b, t = tokens.shape
        prefix = self.state(state).view(b, self.config.prefix_tokens, -1)
        x = torch.cat([prefix, self.tokens(tokens)], dim=1)
        x = self.dropout(x + self.positions[:, : x.shape[1]])
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x[:, self.config.prefix_tokens :]))

    @torch.no_grad()
    def reply(
        self,
        state: torch.Tensor,
        question_ids: list[int],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> list[int]:
        """Generate the answer following ``question <sep>``."""
        self.eval()
        device = next(self.parameters()).device
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state.to(device)
        ids = [BOS] + question_ids + [SEP]
        tokens = torch.tensor([ids], dtype=torch.long, device=device)
        produced: list[int] = []
        for _ in range(max_new_tokens):
            window = tokens[:, -self.config.max_len :]
            logits = self(state, window)[:, -1, :]
            logits[:, PAD] = -float("inf")
            if temperature <= 0:
                nxt = int(logits.argmax(dim=-1))
            else:
                probabilities = (logits / temperature).softmax(dim=-1)
                nxt = int(torch.multinomial(probabilities, 1))
            if nxt == EOS:
                break
            produced.append(nxt)
            tokens = torch.cat(
                [tokens, torch.tensor([[nxt]], dtype=torch.long, device=device)], dim=1
            )
        return produced


class ChatDataset(torch.utils.data.Dataset):
    """Masks the loss to the answer only.

    The model must not be rewarded for predicting the question — that is input, not
    output, and training on it wastes capacity teaching the model to echo.
    """

    def __init__(
        self,
        examples: Sequence[DialogueExample],
        tokenizer: WordTokenizer,
        behaviours: Sequence[str],
        max_len: int = 96,
    ):
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.behaviours = list(behaviours)
        self.max_len = max_len
        if not self.examples:
            raise ValueError("no dialogue examples")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        question = self.tokenizer.encode(example.question, add_special=False)
        answer = self.tokenizer.encode(example.answer, add_special=False)
        ids = [BOS] + question + [SEP] + answer + [EOS]
        ids = ids[: self.max_len + 1]
        inputs, targets = ids[:-1], ids[1:]
        # Everything up to and including <sep> is context: mask it out.
        sep_at = inputs.index(SEP) if SEP in inputs else 0
        masked = [PAD] * sep_at + targets[sep_at:]
        pad = self.max_len - len(inputs)
        return {
            "state": torch.from_numpy(example.state.vector(self.behaviours)),
            "tokens": torch.tensor(inputs + [PAD] * pad, dtype=torch.long),
            "targets": torch.tensor(masked + [PAD] * pad, dtype=torch.long),
        }


@dataclass
class ChatResult:
    epochs: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_perplexity: float = 0.0
    answer_accuracy: float = 0.0
    history: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"loss {self.val_loss:.4f}  perplexity {self.val_perplexity:.2f}  "
            f"exact answers {self.answer_accuracy:.1%}"
        )


def train_chat(
    train_examples: Sequence[DialogueExample],
    val_examples: Sequence[DialogueExample],
    tokenizer: WordTokenizer,
    behaviours: Sequence[str],
    config: ChatConfig | None = None,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 6e-4,
    device: str | torch.device = "cpu",
    verbose: bool = True,
) -> tuple[DogChat, ChatResult]:
    import math

    config = config or ChatConfig(
        vocab_size=len(tokenizer), state_dim=state_dim(len(behaviours))
    )
    torch.manual_seed(0)
    device = torch.device(device)
    model = DogChat(config).to(device)

    train_set = ChatDataset(train_examples, tokenizer, behaviours, config.max_len)
    loaders = {
        "train": torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    }
    if val_examples:
        loaders["val"] = torch.utils.data.DataLoader(
            ChatDataset(val_examples, tokenizer, behaviours, config.max_len),
            batch_size=batch_size,
        )

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * max(1, len(loaders["train"]))
    warmup = max(1, total // 20)
    step = 0
    result = ChatResult()

    for epoch in range(epochs):
        model.train()
        running, seen = 0.0, 0
        for batch in loaders["train"]:
            scale = (step + 1) / warmup if step < warmup else 0.5 * (
                1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))
            )
            for group in optimiser.param_groups:
                group["lr"] = lr * scale
            logits = model(batch["state"].to(device), batch["tokens"].to(device))
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                batch["targets"].to(device).reshape(-1),
                ignore_index=PAD,
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            running += float(loss.detach()) * batch["tokens"].shape[0]
            seen += batch["tokens"].shape[0]
            step += 1

        record = {"epoch": epoch + 1, "train_loss": running / max(1, seen)}
        if "val" in loaders:
            model.eval()
            total_loss, count = 0.0, 0
            with torch.no_grad():
                for batch in loaders["val"]:
                    logits = model(batch["state"].to(device), batch["tokens"].to(device))
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        batch["targets"].to(device).reshape(-1),
                        ignore_index=PAD,
                    )
                    total_loss += float(loss.detach()) * batch["tokens"].shape[0]
                    count += batch["tokens"].shape[0]
            record["val_loss"] = total_loss / max(1, count)
            record["val_perplexity"] = math.exp(min(20.0, record["val_loss"]))
        result.history.append(record)
        if verbose:
            extra = (f"  val_loss={record['val_loss']:.4f} "
                     f"ppl={record['val_perplexity']:.2f}") if "val_loss" in record else ""
            print(f"  epoch {epoch + 1}/{epochs} loss={record['train_loss']:.4f}{extra}",
                  flush=True)

    result.epochs = epochs
    result.train_loss = result.history[-1]["train_loss"]
    result.val_loss = result.history[-1].get("val_loss", 0.0)
    result.val_perplexity = result.history[-1].get("val_perplexity", 0.0)

    if val_examples:
        sample = list(val_examples)[:400]
        matches = 0
        for example in sample:
            ids = model.reply(
                torch.from_numpy(example.state.vector(behaviours)),
                tokenizer.encode(example.question, add_special=False),
            )
            if tokenizer.decode(ids).strip().lower() == tokenizer.decode(
                tokenizer.encode(example.answer, add_special=False)
            ).strip().lower():
                matches += 1
        result.answer_accuracy = matches / max(1, len(sample))
    return model, result


def save_chat(
    path: str | Path,
    model: DogChat,
    tokenizer: WordTokenizer,
    behaviours: Sequence[str],
    result: ChatResult | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": asdict(model.config),
        "vocab": tokenizer.itos,
        "behaviours": list(behaviours),
        "metrics": asdict(result) if result else {},
    }, path)
    return path


def load_chat(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[DogChat, WordTokenizer, list[str], dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = DogChat(ChatConfig(**payload["config"]))
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    return (
        model,
        WordTokenizer(payload["vocab"]),
        payload.get("behaviours", []),
        payload.get("metrics", {}),
    )


@dataclass
class ChatAnswer:
    text: str
    out_of_domain: float
    question: str

    @property
    def uncertain(self) -> bool:
        return self.out_of_domain > 0.4

    def render(self) -> str:
        lines = [f'  you: "{self.question}"', f"  dog: {self.text}"]
        if self.uncertain:
            lines.append(
                "       (a lot of that question was outside what this model was "
                "trained on, so treat the answer with suspicion)"
            )
        return "\n".join(lines)


def ask(
    model: DogChat,
    tokenizer: WordTokenizer,
    behaviours: Sequence[str],
    state: DogState,
    question: str,
    temperature: float = 0.0,
) -> ChatAnswer:
    """Answer one question about a clip's state.

    ``out_of_domain`` is the fraction of question words the tokeniser did not know.
    A conditional decoder will answer *anything* fluently, so a caller needs some
    signal that the question was outside the training distribution — this is the
    cheapest honest one available.
    """
    words = WordTokenizer.split(question)
    unknown = sum(1 for w in words if w not in tokenizer.stoi)
    ids = tokenizer.encode(question, add_special=False)
    reply = model.reply(
        torch.from_numpy(state.vector(behaviours)), ids, temperature=temperature
    )
    return ChatAnswer(
        text=tokenizer.decode(reply),
        out_of_domain=unknown / max(1, len(words)),
        question=question,
    )


def state_from_prediction(prediction) -> DogState:
    """Derive the chat state from a :class:`~dogsai.predict.VideoPrediction`."""
    from .audio import NOT_THE_DOG

    affect = prediction.affect()
    audio = prediction.audio()
    voice_events = [e for e in audio.events if e.kind not in NOT_THE_DOG]
    loudest = max(voice_events, key=lambda e: e.confidence) if voice_events else None
    translation = prediction.translation()
    return DogState(
        behaviour=prediction.dominant() or "standing",
        voice=loudest.kind if loudest else "",
        arousal=float(translation.arousal or affect.arousal),
        valence=float(translation.valence or affect.valence),
    )


def build_tokenizer(examples: Sequence[DialogueExample]) -> WordTokenizer:
    """Vocabulary over both sides of the dialogue, with ``<sep>`` reserved."""
    texts = [e.question for e in examples] + [e.answer for e in examples]
    counts: dict[str, int] = {}
    for text in texts:
        for word in WordTokenizer.split(text):
            counts[word] = counts.get(word, 0) + 1
    vocab = list(CHAT_SPECIALS) + sorted(counts)
    return WordTokenizer(vocab)
