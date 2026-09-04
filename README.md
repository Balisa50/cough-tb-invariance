# Is cough screening learning the disease, or the phone?

Cough-audio classifiers for respiratory disease report strong accuracy and then
fail at a new clinic. In August 2026, Zhang et al. ([arXiv 2608.25846][zhang])
ran the first systematic cross-dataset evaluation and found why: audio
representations organise by **recording device and dataset rather than by
disease status**, predicted probability tracks **country-level prevalence**,
and acquisition variability matters more than population shift. Cross-dataset
AUC falls below 0.6. A model trained in Zambia scored 0.755 at home and 0.581
on CODA.

They diagnosed it. They did not fix it. This repository is the fix, and the
evaluation strict enough to tell whether it worked.

[zhang]: https://arxiv.org/abs/2608.25846

## The claim being tested

> A cough classifier can be made invariant to the recording site, and that
> invariance survives being tested on a country it has never seen.

Two mechanisms, deliberately separable so each can be credited or blamed alone:

- **Device simulation.** Every clip is re-coloured through a randomly sampled
  imaginary microphone: band-limiting, a tilt across the spectrum, two
  resonances, gain, noise floor, occasional saturation. A device that changes
  every epoch cannot identify a site.
- **Adversarial site head.** A second head predicts the site, and its gradient
  is negated on the way back, so the encoder is trained to make the site
  unpredictable from its own features.

## The part that makes it falsifiable

Disease accuracy cannot tell you whether the site was removed. So a fresh
**probe** is trained on frozen features and asked to recover the site.

- probe at chance → the information is gone
- probe still succeeds → the encoder hid the site from its own adversary
  without discarding it, and the invariance claim fails whatever the headline
  number says

Every fold reports `site-leak`, the probe's accuracy above the majority-class
baseline. A gain in accuracy without a fall in site leak is not invariance. It
is a better model that still knows where it is.

## Evaluation

Two rules, enforced in `src/evaluation.py` so they cannot be skipped:

1. **No participant spans a split.** One person contributes many clips.
   Splitting clips at random puts the same larynx on both sides and measures
   memorisation.
2. **The test site is unseen.** Leave-one-site-out is the only split that makes
   reading the site useless.

Reported metric is **specificity at 90% sensitivity**, the WHO triage operating
point, not AUC. Reported alongside is the AUC obtainable from site prevalence
alone, so a model that has learned geography is visible as one. The headline is
the **worst** site, because a screening tool is limited by where it performs
worst.

## Does the evaluation actually work?

Validated on a cohort built with **no disease signal at all**, where the label
is reachable only through site prevalence:

| evaluation | AUC |
| --- | --- |
| same-site random split | **0.634** |
| predicting each site's base rate, ignoring audio | **0.637** |
| leave-one-site-out | **0.499** |

A model that cannot have learned anything scores 0.634 and looks like it works.
That is the published result reproduced where the ground truth is known.

## Running it

```bash
pip install -r requirements.txt
python run_experiment.py --dataset synthetic     # end to end, no data needed
pytest -q                                        # 17 tests
```

With real audio:

```bash
python run_experiment.py --dataset coswara --root data/raw/coswara --meta combined_data.csv
python run_experiment.py --dataset coda    --root data/raw/coda    --meta metadata.csv
```

CPU only. No GPU required.

## Data

Any corpus reduces to one table: `filepath`, `participant_id`, `site`, `label`.
Adding a dataset means writing one function in `src/data.py` and changing
nothing else.

- **CODA TB** ([syn31472953](https://www.synapse.org/Synapse:syn31472953)) is
  the target: 2,143 participants across seven countries, four of them African,
  with microbiological ground truth. Access requires Synapse profile validation,
  which requires an identity attestation on institutional letterhead.
- **Coswara** and **COUGHVID** are openly downloadable and need no permission.
  They are COVID rather than TB and their labels are weaker, but they are
  crowdsourced across thousands of handsets, which makes them a *harder*
  invariance test than CODA's seven controlled sites.

`--dataset synthetic` writes its own WAV files and needs nothing. Those numbers
describe the pipeline, never cough classification, and must not be reported.

## Layout

```
src/evaluation.py   leave-one-site-out, participant grouping, WHO metric
src/audio.py        log-mel features, device simulation
src/models.py       encoder, disease head, adversarial site head, probe
src/data.py         manifest validation and per-dataset adapters
src/train.py        training, the probe audit, baseline vs intervention
run_experiment.py   one command
```

## Status

Method and evaluation are built and tested. Real audio is the only missing
piece. A null result, meaning invariance is unattainable and the modality has a
ceiling, will be reported as clearly as a positive one.
