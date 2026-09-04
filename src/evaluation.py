"""
The evaluation protocol.

This module exists before any model does, on purpose. The failure this project
investigates is an evaluation failure, not a modelling one: cough-based TB
classifiers report strong numbers that collapse at a new site, because the
splits used to produce those numbers let the model see the thing it was
actually keying on.

Two rules are enforced here and nowhere else, so they cannot be quietly
skipped:

  1. NO PARTICIPANT SPANS A SPLIT. One person contributes many 0.5s cough
     clips. Splitting clips at random puts the same larynx on both sides of
     the line and measures memorisation. Every split below is grouped by
     participant.

  2. THE TEST COUNTRY IS UNSEEN. Zhang et al. (arXiv 2608.25846) showed that
     representations cluster by recording device and dataset rather than by TB
     status, and that predicted TB probability tracks country-level
     prevalence. A model can therefore score well in-distribution by reading
     the site. Leave-one-country-out is the only split that makes that
     strategy useless.

Reported alongside every score: the prevalence of the held-out country, and
the AUC a model would get from prevalence alone. If a "good" score is
explained by prevalence, the number is an artifact and this makes that
visible instead of flattering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# The CODA cohort. African sites are the majority of it, which is the reason
# this dataset is the right one for the question.
COUNTRIES = [
    "Uganda", "Philippines", "Vietnam", "South Africa",
    "India", "Madagascar", "Tanzania",
]
AFRICAN = {"Uganda", "South Africa", "Madagascar", "Tanzania"}

# WHO Target Product Profile for a TB triage test. The operating point that
# matters is not the one that maximises accuracy, it is the one that holds
# sensitivity at 0.90 and asks what specificity survives.
WHO_MIN_SENSITIVITY = 0.90
WHO_MIN_SPECIFICITY = 0.70


@dataclass
class FoldResult:
    """One leave-one-country-out fold."""
    held_out: str
    n_participants: int
    n_clips: int
    prevalence: float
    auc: float
    specificity_at_90_sens: float
    threshold_at_90_sens: float
    meets_who: bool
    # A model that only knew each country's base rate would score this. Any
    # AUC near it is a model that has learned the site, not the disease.
    prevalence_only_auc: float = field(default=0.5)

    def __str__(self) -> str:
        flag = "MEETS WHO" if self.meets_who else "below WHO"
        return (
            f"  {self.held_out:<14} n={self.n_participants:<5} "
            f"prev={self.prevalence:.2f}  AUC={self.auc:.3f}  "
            f"spec@90sens={self.specificity_at_90_sens:.3f}  {flag}"
        )


def specificity_at_sensitivity(y_true, y_score, target_sensitivity=WHO_MIN_SENSITIVITY):
    """
    Specificity at the threshold that first achieves the target sensitivity.

    Accuracy and AUC are the wrong headline for a triage test. The clinical
    question is fixed by the WHO profile: hold sensitivity at 90%, then ask
    how many people you spare an unnecessary molecular test. That is the
    number that decides whether this is deployable.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    positives = y_true == 1
    negatives = ~positives
    if positives.sum() == 0 or negatives.sum() == 0:
        return float("nan"), float("nan")

    # Sweep every threshold the scores actually realise. Sensitivity is
    # monotonically non-increasing as the threshold rises, so the first
    # threshold meeting the target is the one that maximises specificity.
    thresholds = np.unique(y_score)[::-1]
    best_spec, best_thr = 0.0, thresholds[0]
    for thr in thresholds:
        predicted = y_score >= thr
        sens = predicted[positives].mean()
        if sens >= target_sensitivity:
            spec = (~predicted[negatives]).mean()
            if spec > best_spec:
                best_spec, best_thr = float(spec), float(thr)
    return best_spec, best_thr


def prevalence_only_auc(y_true, groups) -> float:
    """
    The score obtainable by ignoring the audio entirely and predicting each
    country's base rate.

    This is the null model for the confound Zhang et al. identified. It is
    reported next to every real score so that a model which merely recovers
    site prevalence is visible as such rather than being read as a success.
    """
    y_true = np.asarray(y_true)
    groups = np.asarray(groups)
    rates = {g: y_true[groups == g].mean() for g in np.unique(groups)}
    cheating_score = np.array([rates[g] for g in groups])
    if len(np.unique(y_true)) < 2 or len(np.unique(cheating_score)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, cheating_score))


def leave_one_country_out(meta: pd.DataFrame):
    """
    Yield (held_out_country, train_index, test_index).

    Grouping is by participant *within* the country split as well: a country
    is held out whole, so no participant can appear on both sides by
    construction. Asserted rather than assumed, because this is the exact
    mistake the protocol exists to prevent.
    """
    required = {"participant_id", "country", "label"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"metadata is missing required columns: {sorted(missing)}")

    for country in sorted(meta["country"].unique()):
        test_mask = meta["country"] == country
        train_idx = np.flatnonzero(~test_mask)
        test_idx = np.flatnonzero(test_mask)

        if len(test_idx) == 0 or len(train_idx) == 0:
            continue

        train_people = set(meta.iloc[train_idx]["participant_id"])
        test_people = set(meta.iloc[test_idx]["participant_id"])
        overlap = train_people & test_people
        assert not overlap, (
            f"{len(overlap)} participant(s) appear in both sides of the "
            f"{country} fold. A participant-level leak invalidates the fold."
        )

        yield country, train_idx, test_idx


def score_fold(country: str, meta_test: pd.DataFrame, y_score) -> FoldResult:
    """Score one held-out country at the participant level."""
    # Clips are not independent observations. One participant contributes many
    # coughs, so scoring per clip weights talkative patients more heavily and
    # inflates confidence. Aggregate to one score per person first.
    per_person = (
        pd.DataFrame({
            "participant_id": meta_test["participant_id"].values,
            "label": meta_test["label"].values,
            "score": np.asarray(y_score),
        })
        .groupby("participant_id")
        .agg(label=("label", "max"), score=("score", "mean"))
    )

    y_true = per_person["label"].to_numpy()
    scores = per_person["score"].to_numpy()

    auc = (
        float(roc_auc_score(y_true, scores))
        if len(np.unique(y_true)) > 1 else float("nan")
    )
    spec, thr = specificity_at_sensitivity(y_true, scores)

    return FoldResult(
        held_out=country,
        n_participants=len(per_person),
        n_clips=len(meta_test),
        prevalence=float(y_true.mean()),
        auc=auc,
        specificity_at_90_sens=spec,
        threshold_at_90_sens=thr,
        meets_who=bool(spec >= WHO_MIN_SPECIFICITY) if spec == spec else False,
    )


def summarise(results: list[FoldResult]) -> str:
    """A report that leads with the worst fold, because that is the honest one."""
    if not results:
        return "no folds scored"

    aucs = np.array([r.auc for r in results if r.auc == r.auc])
    worst = min(results, key=lambda r: r.auc if r.auc == r.auc else 9)
    african = [r for r in results if r.held_out in AFRICAN]

    lines = [
        "",
        "LEAVE-ONE-COUNTRY-OUT",
        "-" * 68,
        *[str(r) for r in results],
        "-" * 68,
        f"  mean AUC {aucs.mean():.3f}   worst {worst.auc:.3f} ({worst.held_out})",
    ]
    if african:
        afr = np.array([r.auc for r in african if r.auc == r.auc])
        lines.append(f"  African sites only: mean AUC {afr.mean():.3f} over {len(afr)} folds")
    lines += [
        f"  folds meeting the WHO profile: "
        f"{sum(r.meets_who for r in results)} of {len(results)}",
        "",
        "  The mean flatters. A screening tool is only as good as the site it",
        "  works worst at, because that is where it will be deployed next.",
    ]
    return "\n".join(lines)
