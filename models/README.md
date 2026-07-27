# Shipped checkpoints

## `dogbehaviour-nano.pt`

`nano` preset (0.87 M parameters, 0.15 GMACs per clip at 10x112), trained from
random initialisation — no pretrained weights of any kind — on the
`dogbehaviour` dataset registered in `dogsai/datasets_hub.py`.

```bash
dogsai feeling models/dogbehaviour-nano.pt my_dog.mp4
dogsai predict models/dogbehaviour-nano.pt my_dog.mp4 --overlay annotated.mp4
```

### What it recognises

Five behaviours, and **only** these five:

`chewing` · `eating_drinking` · `eliminating` · `playing` · `yawning`

Anything else your dog does will be forced into one of those five, because a
softmax has no "none of the above" option. A sleeping dog will be reported as one
of these labels with some confidence. That is a limitation of the *training data*,
not a bug: the dataset contains these five behaviours and nothing more.

### Measured performance

244 held-out clips, from 244 source videos disjoint from training (group-aware
split, so no clip and no re-encode of a training clip appears here):

| metric | value | chance |
|--------|------:|-------:|
| balanced accuracy | 0.622 | 0.200 |
| top-1 accuracy | 0.611 | 0.200 |
| macro F1 | 0.607 | — |
| mAP | 0.691 | ~0.200 |

| behaviour | precision | recall | F1 | AP | support |
|-----------|----------:|-------:|---:|---:|--------:|
| eliminating | 0.786 | 0.815 | 0.800 | 0.877 | 54 |
| chewing | 0.419 | 0.750 | 0.537 | 0.668 | 24 |
| playing | 0.536 | 0.577 | 0.556 | 0.668 | 52 |
| yawning | 0.546 | 0.500 | 0.522 | 0.646 | 60 |
| eating_drinking | 0.647 | 0.407 | 0.500 | 0.679 | 54 |

`dogbehaviour-nano.val.json` holds the full evaluation output, including the
per-class thresholds fitted on validation data (also embedded in the checkpoint).

### Honest limitations

- **62% balanced accuracy means it is wrong about a third of the time.** Treat any
  single prediction as a suggestion, not a fact.
- **It had not converged.** Validation was still improving when the epoch budget
  ran out, and it was trained at 112px on CPU. The `small`/`base` presets on a GPU
  should do better; `python scripts/vast_train.py launch` runs that.
- **Eating and yawning are confused** in both directions (a third of eating clips
  are called yawning). At 112px both are open-mouth head motion.
- **Only five behaviours**, none of which are posture or locomotion, so the
  body-language read built on it (`dogsai feeling`) is correspondingly coarse and
  reports low confidence. It says so in its output.
- Trained on other people's footage of other people's dogs, filmed in unknown
  conditions. Your camera angle, lighting, breed and framing are all distribution
  shift.

To do better, the highest-leverage move is not a bigger model — it is annotating
your own footage. `dogsai prepare --format label-studio` and
`dogsai train --resume models/dogbehaviour-nano.pt --resume-weights-only` will
fine-tune this checkpoint onto your data.

## `dogbehaviour-narrator.pt`

`DogNarrator` — a 0.74 M parameter Transformer decoder that turns the detection
vector (behaviour distribution + audio summary + duration) into a sentence.

```python
from dogsai.narrate import load_narrator, narrate
model, tokenizer, behaviours, metrics = load_narrator("models/dogbehaviour-narrator.pt")
```

Validation perplexity 1.04, exact-match caption generation 100%.

**Read that 100% with suspicion.** The training corpus has five distinct caption
strings, one per behaviour, so the task reduces to a five-way lookup from a vector
the model is handed. The number says the pipeline works; it says almost nothing
about the model's language ability. It is included because a real trained
generative model with an honest limitation is more useful than a flattering
benchmark.


## `dogbehaviour-chat.pt`

`DogChat` — 2.16 M param Transformer decoder you can ask free-text questions,
conditioned on a clip's detected state (behaviour, vocalisation, arousal,
valence). Answers in two parts: what the dog would say, and what to do.

```python
from dogsai.chat import ask, load_chat, DogState
model, tokenizer, behaviours, metrics = load_chat("models/dogbehaviour-chat.pt")
state = DogState(behaviour="playing", voice="growl", arousal=0.7, valence=-0.5)
print(ask(model, tokenizer, behaviours, state, "should i be worried").text)
```

Validation perplexity 1.06, 49.5% exact free-generation match against 400 held-out
question/answer pairs (a real number for a paraphrase task, unlike the narrator's
old 100%: there are multiple valid phrasings per state, so exact string match
under-counts correct answers).

**Trained entirely on a self-generated dialogue corpus** (21,600 pairs, `dogsai
train-chat`) — see `dogsai/chat.py`. The advice content is hand-encoded
conservative dog-behaviour guidance, the same body `dogsai.advise` uses; the model
learns to select and phrase it across free-text questions, not to originate new
facts. `ask()` returns an `out_of_domain` score for exactly this reason: a
conditional decoder answers *anything* fluently, including questions far outside
training, and that score is the only signal for when to distrust the answer.
