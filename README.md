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

## GPU training

```bash
python scripts/vast_train.py offers                      # prices, spends nothing
python scripts/vast_train.py launch --preset base --yes   # spends money
python scripts/vast_train.py status
python scripts/vast_train.py fetch-run
python scripts/vast_train.py destroy --yes                # stops billing
```

The instance downloads the dataset itself from HuggingFace rather than waiting on
an upload. Nothing is created without `--yes`. The API key comes from
`$VAST_API_KEY` or `~/.vast_api_key`, both outside the repo — credentials do not
belong in version control, and a key pasted into a chat or a commit should be
rotated.

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
