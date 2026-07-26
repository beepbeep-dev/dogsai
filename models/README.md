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
