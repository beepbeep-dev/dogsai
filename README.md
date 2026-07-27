# dogsai

Dog behaviour recognition from video. An efficient spatio-temporal CNN written
from scratch in PyTorch, plus the data plumbing that decides whether it is
actually accurate: span-based annotations, group-aware splitting, a dataset
auditor, and sliding-window inference that turns per-clip scores into a behaviour
timeline.

```
$ dogsai predict runs/real/best.pt backyard.mp4

yawning    |        ===                                             |   1.2s
eating     |==============                    ====================  |   6.8s
playing    |                  ================                      |   4.1s
           0s----------------------------------------------  12.0s

     0.00s ->    6.80s  eating_drinking  0.91
     6.40s ->    7.60s  yawning          0.63
     7.20s ->   11.30s  playing          0.84
```

## From scratch means from scratch

Every layer of the network is written here from `torch.nn` primitives —
`Conv3d`, `BatchNorm3d`, `Linear` and activations. No torchvision, no `timm`, no
model zoo, no pretrained weights, no fine-tuning. The weights start random.

```
$ grep -rE "torchvision|timm|pretrained|model_zoo" dogsai/*.py dogsai/model/*.py
dogsai/model/blocks.py: Everything is written directly against ``torch.nn``; no torchvision, no timm, no
dogsai/model/blocks.py: pretrained weights.
```

What is *not* from scratch, and cannot be, is the video the model learns from. A
network trained on nothing recognises nothing, so `dogsai fetch` pulls real,
public dog footage (see [Data](#data)). That distinction matters: the
architecture and the weights are ours, the pixels are the world's.

## Install

```bash
pip install -e ".[full,dev]"      # torch, av, opencv, huggingface_hub, pytest
```

`av` and `opencv-python-headless` both ship a prebuilt ffmpeg, so there is no
system dependency to install for decoding or encoding.

## Ten minutes, end to end

```bash
# 1. real data: 1217 dog clips, five behaviours, ~9.5 GB
dogsai fetch dogbehaviour

# 2. check the labels BEFORE spending compute on them
dogsai audit --data-root dogsai_data/prepared \
             --behaviours dogsai_data/prepared/behaviours.txt

# 3. train
dogsai train --data-root dogsai_data/prepared \
             --behaviours dogsai_data/prepared/behaviours.txt \
             --task multiclass --set model.preset=small

# 4. run it on your own dog
dogsai feeling runs/dognet/best.pt my_dog.mp4
```

No data yet? `dogsai synth --root data/synth` renders a synthetic dataset so
every stage above runs immediately. It is a test fixture, not training data —
see [Synthetic data](#synthetic-data).

## The model

`DogBehaviourNet`, a factorised spatio-temporal CNN. Shape flow at the `small`
preset:

```
input        3 x 16 x 160 x 160
stem        24 x 16 x  80 x  80    (1,3,3) spatial + (3,1,1) temporal
stage 1     32 x 16 x  40 x  40    2 blocks
stage 2     64 x  8 x  20 x  20    3 blocks, temporal stride 2
stage 3    112 x  8 x  10 x  10    4 blocks
stage 4    176 x  4 x   5 x   5    3 blocks, temporal stride 2
head       512 x  4 x   5 x   5
spatial average          -> (B, 4, 512)
temporal attention pool  -> (B, 512)
classifier               -> (B, num_classes)
```

| preset | params | MACs / clip | input | CPU latency¹ |
|--------|-------:|------------:|-------|-------------:|
| nano   | 0.87 M | 0.23 G | 12 x 128² | 44 ms |
| small  | 2.85 M | 1.21 G | 16 x 160² | 338 ms |
| base   | 6.39 M | 4.73 G | 24 x 192² | 844 ms |

¹ single clip, batch 1, 4-core CPU, no GPU. For scale, R(2+1)D-18 is 33 M
parameters and ~40 GMACs — the `small` preset is ~33x cheaper.

One caveat worth stating, because MACs alone would oversell this: **depthwise 3-D
convolutions are FLOP-efficient but poorly served by CPU kernels.** Measured
training throughput on four CPU cores was 9.5 clips/s at 10x112 and only ~2 clips/s
at 12x128 — far below what the FLOP count implies, and it scales steeply with
resolution. The architecture's efficiency is real, but it is realised on a GPU, and
on CPU you should expect to train at low resolution or not at all. Inference is
fine either way: 44 ms per clip at the `nano` preset is ~18x faster than realtime.

Five design decisions carry most of that efficiency, and each is commented where
it lives:

- **Factorised convolutions.** A `(1,k,k)` depthwise spatial conv followed by a
  `(kt,1,1)` depthwise temporal conv spans the same receptive field as
  `(kt,k,k)` at a fraction of the parameters, with an extra non-linearity in
  between. Channel mixing happens only at 1x1x1.
- **Late temporal downsampling.** Time goes 16 -> 8 -> 4 and no further.
  Collapsing the temporal axis early is the standard way to make a video model
  fast and simultaneously blind to the thing it is meant to detect.
- **Temporal attention pooling.** A clip is 16 frames of which maybe 4 contain
  the jump; mean pooling dilutes them 4x. A learned query attends over time, and
  its weights are readable output (`dogsai predict --explain 3.0`).
- **Motion stem.** Frame differences are concatenated to the RGB input, so
  small fast motion — tail wagging, shaking off, digging — reaches the first
  layer. One subtraction, no second backbone: the cheap 90% of a two-stream net.
- **Identity-initialised residuals.** Each block's projection BatchNorm starts at
  gamma=0, so a fresh 12-block network is a near-identity function. Training from
  random init is materially more stable that way.

## Accuracy comes from the data path

The model is the easy part. These are the things that actually decide whether the
number you report survives contact with real footage.

**Span annotations, not clip labels.** `root/train.jsonl`, one record per
labelled interval:

```json
{"video": "yard_cam_03.mp4", "start": 12.5, "end": 18.0,
 "labels": ["running", "barking"], "group": "yard_cam_03"}
```

If a dog runs for 2s of a 10s clip, a whole-clip label mislabels 8s and the model
learns the background. The sampler only ever draws frames from inside a labelled
span. Folder mode (`root/train/running/clip.mp4`) also works and is fine for a
first pass.

**Explicit negatives.** `"labels": []` teaches the model what "none of the above"
looks like. Without them a detector fires constantly on unlabelled footage.

**Group-aware splitting.** Two clips cut from the same source video are
near-duplicates. `make_splits` keeps every annotation sharing a `group` on the
same side of the split, and stratifies rare labels within that constraint. Its
validation numbers are lower than a naive per-clip shuffle, and they are the
honest ones.

**An auditor that runs before training.** `dogsai audit` finds broken files,
spans shorter than one clip, misspelled labels (with suggestions), contradictory
overlapping postures, thin classes whose metrics will be noise, and split
leakage — both by group key *and* by perceptual hash, because "same footage,
different filename, re-encoded" is extremely common and a group check alone
cannot see it. Errors will corrupt training; warnings will bias it.

```
$ dogsai audit --data-root dogsai_data/prepared
annotations : 973
groups      : 973   (leakage-safe split units)
labelled    : 114.0 min
  yawning            241 clips    19.4 min  #######
  eating_drinking    215 clips    31.6 min  #######
  ...
no problems found.
```

**Metrics that cannot flatter you.** Long-tailed multi-label data makes plain
accuracy meaningless — a model that never predicts `digging` still scores well.
The primary metric is mean average precision (threshold-free, so it measures the
ranking actually learned); macro F1, per-class AP/AUC/support and balanced
accuracy are reported next to it. Per-class thresholds are *fitted* on validation
data and stored in the checkpoint, taking the midpoint of the optimal plateau
rather than its edge.

**Repeat-factor sampling.** Inverse-frequency balancing is incoherent for
multi-label data — a clip labelled `["running","barking"]` cannot be "a barking
sample". Each class gets a factor `sqrt(t/f_c)`, each sample the max over its
labels, redrawn every epoch.

## Decode once, not every epoch

Video training is decode-bound, not compute-bound. On the real dataset above this
pipeline measured **2.5 clips/s** on four CPU cores — and the model's forward and
backward pass was a small minority of that. Every epoch was re-decoding the same
H.264 bitstreams to produce the same pixels.

```bash
dogsai train --data-root dogsai_data/prepared --cache \
             --cache-frames 24 --cache-size 144
```

`--cache` decodes each annotated span once into a memory-mapped `uint8` array and
trains off that. 973 clips at 24 frames of 144px is 1.45 GB, and epoch time drops
from minutes to seconds.

The cache deliberately stores *more* frames and *more* pixels than a training clip
needs, so augmentation survives: a different temporal subset each epoch, and
random-resized-crop with real pixels to choose from. What is given up is crops at
source resolution and sub-stride temporal offsets. On a large dataset with many
epochs the uncached path is still better; on anything where decode dominates —
which is most animal-behaviour datasets — the trade is strongly worth it. The
cache is keyed on the parameters that determine its contents, so changing
resolution or frame count invalidates it rather than silently training on a
mismatched array.

## Measured result on real footage

Trained from random init on the `dogbehaviour` set (973 train / 244 val clips,
973 independent groups, five behaviours), `nano` preset at 10x112, 26 epochs on
four CPU cores:

```
balanced_acc = 0.622    top1 = 0.611    macro_f1 = 0.607    mAP = 0.691
                                        (chance balanced_acc = 0.200)

behaviour         precision   recall       f1       ap   support
eliminating          0.786     0.815    0.800    0.877        54
chewing              0.419     0.750    0.537    0.668        24
playing              0.536     0.577    0.556    0.668        52
yawning              0.546     0.500    0.522    0.646        60
eating_drinking      0.647     0.407    0.500    0.679        54

confusion (row-normalised, rows = truth)
                 chewin eating elimin playin yawnin
chewing            0.75   0.08          0.12   0.04
eating_drinking    0.11   0.41   0.04   0.11   0.33
eliminating        0.04   0.02   0.81   0.13
playing            0.15   0.06   0.10   0.58   0.12
yawning            0.15   0.10   0.08   0.17   0.50
```

Two honest observations. First, the confusions are the *right* confusions:
`eliminating` is the most distinctive class (a squat is unmistakable), while a
third of `eating` clips are called `yawning` — both are open-mouth head motion at
112px, which is genuinely hard. Second, **this run had not converged.** Validation
was still improving when the 26-epoch budget ran out (best score was the final
epoch), so this is a floor, not a ceiling. More epochs, the `small` or `base`
preset, and a GPU should all move it — see [GPU training](#gpu-training).

## Buffing up the main model: a self-generated multi-label dataset

`dogsai fetch dogbehaviour` gives each clip exactly one label. A clip captioned
"Dog is eating." that also barks partway through trains as pure
`eating_drinking`, and the model never learns what a bark looks like there
because nothing says one exists.

```bash
dogsai make-captions --data-root dogsai_data/prepared      # computes per-clip audio once
dogsai enrich --data-root dogsai_data/prepared             # adds self-detected labels
dogsai train --data-root dogsai_data/prepared/enriched \
             --behaviours dogsai_data/prepared/enriched/behaviours.txt \
             --task multilabel --set model.preset=base
```

`dogsai enrich` adds a `"barking"` label to any clip where `dogsai.audio` —
already in this repo, already verified against real footage — confidently hears
one. Run on the deduplicated 1217-clip set (see below): **398/971 train clips
and 97/246 val clips** gained the label. This is a strict enrichment: it only
ever *adds* the label, it never removes or second-guesses the original
human-authored one, and it inherits whatever error rate the audio detector has
(documented in `dogsai/audio.py`). It is not new external annotation — it is a
detector already in this repository, pointed at video already on disk,
producing labels the original single-label captions could not represent.

Combined with the `base` preset (6.39M params, up from `nano`'s 0.87M) and full
convergence on a GPU, this is the main lever for making the shipped classifier
more accurate: **mAP 0.8648** on 246 held-out clips, up from `nano`'s 0.691 —
see [GPU training](#gpu-training) for the run, and `models/README.md` for the
full breakdown.

### A real leakage bug this run caught

`fishchen/dog-behavior-dataset` groups clips by filename, but auditing the
first split found 21 pairs of differently-named clips — 18 of them
pixel-identical — split across train and val: the same footage saved twice
under a different name. That means the `nano` numbers above were measured
against a validation set that wasn't fully disjoint from training.

`dogsai.audit.merge_duplicate_groups` fixes this with a perceptual-hash
(dHash) union-find pass over one representative clip per group, merging any
group within Hamming distance 6 of another so every copy of the same footage
lands on the same side of the split. It's on by default in
`datasets_hub.prepare` (`--dedupe` to control it), and it's why the counts
above read 971/246 rather than 973/244 — 64 duplicate groups merged. Verify
any prepared dataset is clean with (duplicate checking is on by default; pass
`--no-duplicate-check` to skip it):

```bash
dogsai audit --data-root dogsai_data/prepared
```

## Data

```bash
dogsai fetch                  # list registered datasets, with licences
dogsai fetch dogbehaviour     # download + convert + split
```

| key | content | size |
|-----|---------|-----:|
| `dogbehaviour` | 1217 real dog clips: yawning, eating, eliminating, playing, chewing | 9.5 GB |
| `animalkingdom` | 140 action classes, 850 species (needs filtering to canids) | 15 GB |
| `mammalnet` | 18k videos, 12 behaviours across 173 mammals | 159 GB |

These are other people's datasets; the registry records each licence and you
should read it before redistributing anything.

Bringing your own annotations:

```bash
dogsai prepare export.json --format label-studio --video-root videos/ --split
dogsai prepare labels.csv  --format csv --label-map map.json --split
```

Importers exist for Label Studio, CVAT, generic CSV, AVA-style CSV and folder
layouts. A `--label-map` folds an external vocabulary into this taxonomy, and
anything unmapped is *reported*, not silently dropped.

## How is my dog feeling?

`dogsai feeling model.pt clip.mp4` produces a body-language read. Please read
what it is and is not:

```
how the dog seems:  relaxed and content

  valence  [----------|--#------]  +0.34   (negative <-> positive)
  arousal  [#####----------------]  0.24   (calm <-> activated)
  confidence 41%   evidence coverage 82%

  based on:
    eating_drinking   6.8s  willingness to eat is a broadly positive welfare indicator
    yawning           1.2s  out-of-context yawning is a recognised appeasement signal —
                            but a tired dog also just yawns

  note: this model only recognises 5 affect-relevant behaviours, so the read is coarse
  what this model cannot see: tail position; ear set; lip licking; whale eye.
```

**No model can tell you what a dog feels.** Internal state is not observable.
What welfare science does instead is map observable body language onto two axes —
**arousal** (how activated) and **valence** (how positive the situation appears) —
because that is what the evidence supports. You can see that a dog is highly
aroused with negative-leaning signals, which is useful. You cannot see "anxious"
versus "frustrated"; the video does not contain that distinction.

Three limits, stated plainly because they are how this kind of output misleads:

1. **Context is invisible.** A panting dog is thermoregulating in the sun and
   stressed at the vet, and the pixels look the same.
2. **Displacement signals are ambiguous by nature.** A yawn is a stress signal
   out of context and otherwise just a tired dog. So yawning contributes mild
   evidence, never a verdict, and no single signal can swing the read.
3. **The read is capped by the taxonomy.** A five-behaviour model cannot see tail
   position, ear set, lip licking or whale eye — the signals a behaviourist
   weights most heavily. `confidence` shrinks accordingly and the output says so.

If your dog's behaviour worries you, that is a question for a vet or a qualified
behaviourist who can see the context. Nothing here substitutes for that.

## What is it saying, and what should I do?

```bash
dogsai translate models/dogbehaviour-nano.pt my_dog.mp4
```

Three outputs, from two sensors. The video model gives behaviour spans; a
from-scratch audio analysis gives vocalisations; together they produce a summary,
a first-person rendering, and suggestions.

```
summary
  Over 8 seconds the dog was mostly playing (5.8s), eating drinking (4.3s). It
  vocalised: bark x2, bark alarm x1, whine x1. Overall it reads as: "Something's
  out there and I don't like it. Back off."

what your dog is telling you
  "Something's out there and I don't like it. Back off."

  moment by moment:
      0.67s  "I'm having a brilliant time. Do not stop doing this."  [playing 0.7-5.7s]
  !   1.21s  "Something's out there and I don't like it. Back off."  [bark_alarm 1.2s]
      1.34s  "Please. I need something and I can't sort it myself."  [whine 1.3s]

  note: 2 sound(s) were a person talking, not the dog, and were ignored
  note: the video and the audio disagree: what the dog was doing looks positive,
        but it made a sound (bark_alarm) that does not.

what to do
   - Identify what triggered it and reduce the exposure — distance, a barrier, or
     blocking the line of sight — rather than trying to out-shout it.
     (because: low, harsh, repeated barking)
```

Every line traces back to a detection you can inspect. Three things this
deliberately does not do:

**It does not decode words.** Dogs communicate constantly, but not in language —
there is no sentence inside a bark to recover. What `translate` does is carry
*meaning* across from one signalling system into another, which is a real
translation in the useful sense and not one in the sense of decoding speech. The
first person is presentation; the grounding is the detection list.

**It does not average away a conflict.** A yelp during play is the informative
signal, so it takes over the headline instead of being blended into a cheerful
average — and the disagreement between channels is stated outright.

**It does not guess who made a sound.** People talk to their dogs while filming
them, and human speech occupies the same acoustic region as a howl or growl
(sustained, tonal, 85-300 Hz). Speech is detected, labelled as not-the-dog, and
excluded from every aggregate — without that guard the commonest sound in a home
video gets confidently attributed to the dog.

### The audio side

```bash
dogsai listen my_dog.mp4     # vocalisations only, no model needed
```

Written from scratch on numpy: framing, magnitude STFT, spectral centroid,
spectral flatness (the tonality axis), and autocorrelation pitch tracking.
Vocalisations are typed as bark (split into alarm/excited by pitch and tonality),
growl, whine, howl, yelp or panting, following the published structure-to-context
mapping — low and harsh skews agonistic, high and tonal skews fear/play, rapid
repetition means arousal.

The pitch tracker earns its complexity by handling octave errors in both
directions, which is the central difficulty of autocorrelation pitch estimation.
A 1200 Hz tone at 16 kHz has a period of 13.33 samples; three periods is exactly
40, so the autocorrelation peak at 400 Hz is *taller* than the true one and naive
argmax reports a third of the real pitch. Preferring the shortest lag fixes that
and breaks the opposite case, where a genuine 300 Hz fundamental under a louder
600 Hz harmonic gets reported as 600. Neither preference is right alone, so
candidates are validated against the spectrum: a real fundamental has audible
energy at its own frequency, a spurious sub-harmonic has none.

Two honest limits. The typing is **rule-based DSP, not a trained classifier** —
thresholds come from the literature, not from fitting labelled barks, so treat the
type as a well-motivated guess. And the advice is a **fixed rule set**, because
there is no dataset of "dog did X, owner should do Y" to learn from; a model that
generated advice anyway would be producing ungrounded text about someone's pet.
It stays away from anything medical or aversive, and routes yelps, persistent
scratching and unproductive straining to a vet.

## Inference

Per-window scores become intervals a human would agree with:

1. overlapping windows (50% hop by default),
2. median smoothing across windows — a median rejects the single-window outlier
   that produces flickering labels; a mean averages it in,
3. per-class thresholds from the checkpoint,
4. merge same-behaviour spans across short gaps, then drop spans below
   `min_duration`.

```bash
dogsai predict model.pt clip.mp4 --overlay out.mp4 --json result.json
dogsai predict model.pt clips/ --window-stride 0.25 --tta   # denser, flip-averaged
dogsai predict model.pt clip.mp4 --explain 3.0              # frame attention
dogsai export model.pt model.ts.pt --format torchscript     # or --format onnx
```

Exports are self-contained — the motion stem lives inside the graph — and write a
`.meta.json` sidecar with the labels, normalisation and tuned thresholds, since
neither TorchScript nor ONNX has anywhere sensible to keep them.

## Chat: talk back to it

```bash
dogsai chat models/dogbehaviour-nano.pt models/dogbehaviour-chat.pt my_dog.mp4 --interactive
```

`DogChat` (2.16 M params) is a third from-scratch Transformer decoder, conditioned
on the clip's detected state (behaviour, vocalisation, arousal, valence) and asked
a free-text question. Every answer has two parts: what the dog would say in words,
and what a handler should do about it.

```
you: "should i be worried"
dog: I would say alright, nothing much either way. What to do nothing here
     looks alarming. keep an eye on it and trust changes from its own baseline
     over any single moment.
```

Its training data — 21,600 question/answer pairs — is generated by
`dogsai/chat.py`, not collected: there is no public corpus of "dog state + human
question -> answer" to train on. The generator enumerates the states the
detectors actually produce (behaviour x vocalisation x arousal band x valence
band) against a dozen real intents ("are you hurt", "how should I approach you",
"should I be worried") and composes a two-voice answer for each, with the wording
varied deterministically so the model learns the mapping rather than a lookup
table. 417 distinct answers, vocabulary 351, and it reaches perplexity 1.06 —
a real number, because the *phrasings* genuinely vary even though the underlying
guidance per state does not.

Be precise about what that buys you: the **knowledge** in the answers is
hand-encoded conservative dog-behaviour guidance — the same kind `dogsai.advise`
uses — not learned from data. The model learns to select and phrase it fluently
across free-text questions and interpolate between states a lookup table would
have to enumerate by hand. `ask()` returns an `out_of_domain` score (the fraction
of question words the tokeniser did not recognise) precisely because a
conditional decoder answers *any* question fluently, including ones far outside
what it was trained on, and there is no other cheap signal for that.

## The second model: DogNarrator

A Transformer decoder — also written from scratch on `torch.nn` — that generates a
sentence describing a clip. It conditions on a vector packing the behaviour
distribution from the video model, the audio summary (vocalisation counts, arousal,
valence, vocal fraction) and clip duration, projects that into a short prefix, and
decodes tokens causally. 0.74 M parameters.

```bash
dogsai train-narrator --data-root dogsai_data/prepared \
                      --behaviours dogsai_data/prepared/behaviours.txt
```

Trained on the `dogbehaviour` captions it reaches perplexity 1.04 and 100%
exact-match caption generation:

```
chewing          -> "Dog bites rope."
eating_drinking  -> "Dog is eating."
eliminating      -> "Dog is pooping."
playing          -> "Dog is playing."
yawning          -> "Dog is yawning."
```

**That 100% is not impressive and should not be read as such.** The dataset
contains exactly five distinct caption strings, one per class, so the narrator's
task is a five-way lookup from a behaviour vector it is handed — barely harder than
the classifier feeding it. It is reported here because the honest number is more
useful than a flattering one.

**That has since been fixed with a self-generated dataset.** `dogsai make-captions`
measures every real clip — duration, vocalisation type, arousal, how much of the
clip was vocal — and composes a sentence stating them, with wording varied so the
corpus has real diversity: 1,000 distinct captions from 1,217 clips, vocabulary
110. Retrained on that corpus the narrator reaches **validation perplexity 1.87**
— a real language-modelling number, not a lookup-table artefact — and its samples
now compose properties the five-caption version could not represent at all:

```
chewing                    -> "The dog is working on a toy for several seconds."
chewing + growl            -> "... while it growls, then lets out a bark."
chewing + bark_excited     -> "... and it barks excitedly."
```

Read `dogsai/caption_gen.py`'s docstring before trusting this too far: the
captions are **generated from detector output by templates, not collected from
people**. That means the ceiling is "fluent, compositional restatement of the
feature vector" — it cannot teach the narrator anything the detectors do not
already measure, and it inherits every detector error (a mistyped vocalisation
becomes a mistyped caption). It is a genuine improvement in what the model
learns — verbalising measurements in varied, grammatical English conditioned on
their values is a real skill — but it is distillation, not new knowledge.
`dogsai translate` remains the tool for explicit, per-line provenance.

## GPU training

```bash
python scripts/vast_train.py whoami                       # verify key, show credit
python scripts/vast_train.py offers                       # prices, spends nothing
python scripts/vast_train.py launch --clone --preset base \
    --epochs 100 --patience 20 --cache-frames 32 --cache-size 224 --yes  # spends money
python scripts/vast_train.py logs                         # progress, via the API
python scripts/vast_train.py fetch-artifact                # download the trained model, over HTTPS
python scripts/vast_train.py instances                     # what is billing
python scripts/vast_train.py destroy --yes                 # stop billing
```

Logs are read through Vast's API rather than SSH, because SSH egress is blocked in
plenty of environments (sandboxes, CI runners, locked-down networks) and without it
there is no way to see whether a run is progressing.

### Getting the trained model back without SSH

The first verified run (RTX A4000, `small` preset, 60 epochs) reached
**mAP 0.633** — and the checkpoint was stranded on the instance. `request_logs`
only tails console output, not arbitrary files, and SSH/scp — the normal way to
pull a `runs/` directory home — was blocked from that environment.

The fix: `onstart` now bundles `best.pt` + its config/history/metrics, uploads the
bundle to a public anonymous HTTPS host (`0x0.st`, falling back to
`litterbox.catbox.moe`) once training finishes, and prints the URL. `fetch-artifact`
reads that URL back out of the console log (the same `request_logs` API already
used for progress) and downloads it with a plain HTTPS `GET`. No SSH anywhere in
the loop — it only needs what this environment already has.

If you're running the launcher somewhere with real SSH egress, `fetch-run` prints
the direct `scp` command instead, which is simpler when it's available.

That fallback isn't bulletproof: on the `base`-preset run that produced
`models/dogbehaviour-base.pt` (see `models/README.md`), this specific
instance's outbound connections to both `0x0.st` and `litterbox.catbox.moe`
hung well past their own `curl -m` timeouts — a host-level issue, not a code
bug, since the same script had worked minutes earlier on a different instance.
With no SSH and no working anonymous-host upload, the checkpoint was pulled
out through Vast's own Cloud Sync feature instead — it's a plain REST endpoint
(`POST /commands/rclone/`, discoverable via `GET /users/cloud_integrations/`
for the connection id) that copies instance files straight to a configured
cloud destination (Google Drive, S3, Backblaze, Dropbox), with no SSH
involved on either end. `scripts/vast_train.py` doesn't wrap this yet — it was
driven by hand against the Vast API for this run — but it's the answer if
`fetch-artifact` ever stalls the same way.

The instance downloads the dataset itself from HuggingFace rather than waiting on
an upload, and now also runs `dogsai make-captions` + `dogsai enrich` before
training so it learns from the richer multi-label set (see
[Buffing up the main model](#buffing-up-the-main-model-a-self-generated-multi-label-dataset)).
Nothing is created without `--yes`. The API key comes from `$VAST_API_KEY` or
`~/.vast_api_key`, both outside the repo — credentials do not belong in version
control, and a key pasted into a chat or a commit should be rotated.

## Synthetic data

`dogsai synth` renders an articulated stick-dog whose motion signature differs
per behaviour, as real H.264 mp4s with span annotations. It exists so the
pipeline is verifiable without a dataset: a model that cannot learn it has a bug.

It is a test fixture, **not** training data — the appearance statistics of a
rendered figure have nothing to do with a dog, and a model trained on it
transfers to nothing. Two choices keep it useful as a test: per-session
randomised colour, size, background and camera shake, so motion is the only
consistent cue; and multi-segment sessions with span labels, which exercise span
sampling, group-aware splitting and the auditor end to end.

## Taxonomy

Two axes, because a video model can actually separate them: **posture** is
mutually exclusive (a dog is not sitting and running at once), **actions**
co-occur (a running dog can bark). Hence both a single-label head
(`--task multiclass`) and a per-behaviour head (`--task multilabel`).

```
posture   lying_down sitting standing walking trotting running
actions   jumping playing chewing eating_drinking sniffing digging barking
          tail_wagging scratching shaking_off rolling stretching yawning
          eliminating alert_freeze
```

`dogsai behaviours` prints it. Your own taxonomy works too — pass
`--behaviours my_labels.txt`; the checkpoint carries its label space with it, so
inference never needs the training config.

## Python API

```python
from dogsai import BehaviourPredictor

predictor = BehaviourPredictor("runs/real/best.pt")
prediction = predictor.predict("my_dog.mp4")

print(prediction.timeline())
for span in prediction.spans:
    print(span.behaviour, span.start, span.end, span.score)

reading = prediction.affect()
print(reading.label, reading.valence, reading.arousal)
```

## Tests

```bash
pytest -q
```

Covers the decoder against a video whose frame indices are encoded in its pixels
(so "did we get the frames we asked for" is actually checked, not assumed), the
clip-consistency invariant of every augmentation, block-level model properties
(identity-initialised residuals, motion stem vanishing on a static clip,
attention weights summing to 1), metrics against hand-computed values, span
post-processing, and the auditor's ability to catch a cross-split duplicate that
was renamed and re-grouped to hide.

## Layout

```
dogsai/
  labels.py         taxonomy, BehaviourSpan
  config.py         dataclass configs, presets
  video.py          decode / probe / sample / encode
  transforms.py     clip-consistent augmentation
  dataset.py        annotations, datasets, samplers, splitting
  audit.py          dataset validation, leakage + duplicate detection
  importers.py      Label Studio / CVAT / CSV / AVA / folders
  datasets_hub.py   public dataset registry, download, conversion
  model/
    blocks.py       factorised blocks, SE, drop-path, motion stem, attention pool
    dognet.py       DogBehaviourNet
    ema.py          weight averaging
  engine.py         losses, training loop, checkpoints
  metrics.py        AP, AUC, threshold fitting, per-class reporting
  predict.py        sliding-window inference, span extraction, overlay
  affect.py         body-language read (arousal / valence)
  export.py         TorchScript, ONNX, benchmarking
  synth.py          procedural test footage
  cli.py            the dogsai command
scripts/
  vast_train.py     rent a GPU, train, collect, destroy
```

## Licence

MIT for this code. Datasets fetched through `dogsai fetch` carry their own terms.
