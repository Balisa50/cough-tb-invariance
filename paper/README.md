# Paper

**Site-invariant representations for cough-audio screening: removing the
confound does not recover disease signal.** Abdoulie Balisa, September 2026.

- [`Balisa-2026-site-invariant-cough-screening.pdf`](Balisa-2026-site-invariant-cough-screening.pdf) — 8 pages, A4
- [`paper.html`](paper.html) — the source the PDF is rendered from

## Summary

Adversarial domain-invariance removes recording-site information from cough
representations, confirmed by a probe trained on frozen features to recover the
site: leakage falls from 0.018 to zero in eight of nine folds, Wilcoxon
`p = 0.012`. Disease classification does not change, mean AUC 0.464 to 0.489,
`p = 0.359`, because the corpus holds no cross-country signal to preserve. A
null model predicting each country's base rate while ignoring the audio scores
`0.741`, above every fold measured.

## Rebuilding the PDF

```bash
python paper/build_pdf.py
```

Requires Chrome, `pypdf` and `reportlab`. The script inlines the three
typefaces as base64 so embedding does not depend on a network fetch at print
time, forces the light palette since a PDF has no viewer theme, and passes
`--no-pdf-header-footer`. That last flag matters: without it Chrome stamps every
page with a timestamp and the local file path, which leaks the author's
directory structure into a document meant to be circulated.

Fonts are pulled from Google Fonts into `fonts-inline.css` on first build.

## Numbers

Every figure and table is generated from
[`../results/coughvid_folds.json`](../results/coughvid_folds.json), the direct
output of `scripts/run_resumable.py`, rather than transcribed by hand. Regenerate
the summary tables with:

```bash
python scripts/report_results.py results/coughvid_folds.json
```

If the paper and that command ever disagree, the JSON is correct and the paper
is stale.
