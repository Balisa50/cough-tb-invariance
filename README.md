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
python run_experiment.py --dataset coughvid --root data/raw/coughvid/public_dataset
python run_experiment.py --dataset coswara  --root data/raw/coswara --meta combined_data.csv
python run_experiment.py --dataset coda     --root data/raw/coda    --meta metadata.csv
```

A full leave-one-country-out sweep is eighteen CPU folds and takes hours, so
`scripts/run_resumable.py` writes each fold to disk as it finishes and skips
what is already there. Re-running continues rather than restarting.

CPU only. No GPU required.

## Data

Any corpus reduces to one table: `filepath`, `participant_id`, `site`, `label`.
Adding a dataset means writing one function in `src/data.py` and changing
nothing else.

- **CODA TB** ([syn31472953](https://www.synapse.org/Synapse:syn31472953)) is
  the target: 2,143 participants across seven countries, four of them African,
  with microbiological ground truth. Access requires Synapse profile validation,
  which requires an identity attestation on institutional letterhead.
- **COUGHVID** ([Zenodo 4048312](https://zenodo.org/records/4048312), CC-BY-4.0)
  is downloaded by `scripts/fetch_coughvid.py` and needs no permission. It is
  the corpus the result above is measured on: 20,072 crowdsourced submissions,
  of which 2,739 carry a label, an actual cough and a location. COVID rather
  than TB, and self-reported, but recorded on thousands of different handsets,
  which makes the device shift real rather than a proxy for it.
- **Coswara** is likewise open, and its adapter is written but unexercised.

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

## Result: COUGHVID

2,739 recordings, 9 countries, COVID-19 against healthy, leave-one-country-out.
Every fold holds out one country entirely.

| held out | clips | prev. | base AUC | base sp@90 | base leak | interv. AUC | interv. sp@90 | interv. leak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AR | 96 | 0.19 | 0.368 | 0.141 | +0.029 | 0.431 | 0.051 | -0.000 |
| BR | 110 | 0.05 | 0.392 | 0.124 | +0.011 | 0.396 | 0.210 | +0.001 |
| CH | 201 | 0.09 | 0.428 | 0.033 | +0.026 | 0.368 | 0.093 | +0.001 |
| ES | 605 | 0.08 | 0.519 | 0.193 | +0.012 | 0.469 | 0.116 | -0.000 |
| FR | 115 | 0.10 | 0.523 | 0.077 | +0.026 | 0.637 | 0.510 | -0.007 |
| IR | 63 | 0.25 | 0.523 | 0.043 | +0.010 | 0.592 | 0.021 | -0.000 |
| TR | 830 | 0.04 | 0.549 | 0.121 | +0.043 | 0.509 | 0.113 | +0.000 |
| US | 231 | 0.03 | 0.449 | 0.120 | +0.004 | 0.427 | 0.231 | +0.006 |
| UZ | 488 | 0.29 | 0.420 | 0.017 | -0.000 | 0.571 | 0.207 | -0.000 |

|  | baseline | intervention | Wilcoxon p |
| --- | --- | --- | --- |
| mean AUC | 0.464 | 0.489 | 0.359 |
| mean specificity @ 90% sens | 0.096 | 0.173 | 0.301 |
| mean site leak | +0.018 | -0.000 | **0.012** |
| **predicting country base rate, ignoring audio** | **0.741** | **0.741** | |

Two findings, and they point in opposite directions. Keeping them apart is the
reason the probe exists.

**The invariance works.** Site leak falls from +0.018 to zero, in 8 of 9 folds,
p = 0.012. A probe trained on frozen features to recover the recording country
cannot beat its majority-class baseline. This is not inferred from accuracy; it
is measured by an adversary built to break the claim.

**There is no cough signal here to protect.** Mean AUC across unseen countries
is 0.489 with the intervention and 0.464 without, both indistinguishable from
chance, and the change is not significant. Meanwhile predicting each country's
base rate while ignoring the audio entirely scores **0.741**. Geography is worth
more than the cough by a wide margin.

Specificity at 90% sensitivity rises from 0.096 to 0.173, which is a large
relative move and still nowhere near the 0.70 a WHO triage tool requires. AUC
and the operating point also disagree per fold: Iran gains AUC (0.523 to 0.592)
while its specificity halves (0.043 to 0.021). That divergence is why the
headline metric here is specificity at a fixed sensitivity rather than AUC.

Turkey is the clearest single case. It is the strongest baseline fold (0.549)
and the leakiest (+0.043). Remove the site information and it falls to 0.509.
The performance was the shortcut.

### What this does and does not establish

It does not test TB, and COUGHVID cannot. Labels are self-reported COVID status,
only 4% of clips carry an expert diagnosis, and 372 positives across nine
countries is thin. A negative result on this corpus is a statement about this
corpus and about crowdsourced cough audio, not about cough screening for
tuberculosis.

Three limitations are structural rather than incidental:

- **No participant identifiers.** Each uuid is one anonymous submission, so
  grouping is per recording. A repeat submitter is undetectable, and the
  guarantee weakens from "no participant spans a split" to "no recording does".
- **Country is a proxy for device, not a measurement of it.** Two countries can
  share a handset market. The site probe reports how separable the groups
  actually were, which is the honest way to carry that uncertainty.
- **Location is present on 58% of rows**, so the usable set is 2,739 of 20,072.

What it does establish is that the method removes site information on real
audio, and that the evaluation is strict enough to show when a corpus has
nothing underneath. On a within-site split these numbers would have looked
respectable. Held out to an unseen country, they collapse below the null.

## Status

Method, evaluation and one real-corpus result are complete. The README promised
that a null result would be reported as clearly as a positive one, and the
COUGHVID result above is that null.

What is missing is a corpus with microbiological ground truth rather than
self-report. CODA TB ([syn31472953](https://www.synapse.org/Synapse:syn31472953))
remains the target: 2,143 participants, seven countries, four African, with
culture-confirmed labels. Access requires Synapse profile validation, which
requires an identity attestation on institutional letterhead. That request is
in progress.

Reproduce with:

```bash
python scripts/fetch_coughvid.py                     # 951 MB, checksum verified
python scripts/run_resumable.py --root data/raw/coughvid/public_dataset
```
