"""
The protocol must catch the failure it exists to catch.

These tests run on synthetic data built to contain the confound Zhang et al.
measured in the real cohort. The claim under test is not "the model is good".
It is "this evaluation tells the truth about the model", which is the thing
that was missing from the published results.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import (  # noqa: E402
    leave_one_country_out, prevalence_only_auc, score_fold,
    specificity_at_sensitivity,
)
from src.synthetic import make_cohort  # noqa: E402


@pytest.fixture(scope="module")
def cohort():
    return make_cohort(seed=0)


def _fit_predict(X_tr, y_tr, X_te):
    model = LogisticRegression(max_iter=2000)
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def _person_level_auc(meta_test, scores):
    """Aggregate clips to one score per participant before scoring."""
    frame = pd.DataFrame({
        "pid": meta_test["participant_id"].to_numpy(),
        "y": meta_test["label"].to_numpy(),
        "s": np.asarray(scores),
    })
    grouped = frame.groupby("pid").agg(y=("y", "max"), s=("s", "mean"))
    return roc_auc_score(grouped["y"], grouped["s"])


def test_a_model_with_nothing_to_learn_still_looks_skilful_within_sites():
    """
    The core demonstration, on a cohort containing no disease signal at all.

    With `disease_strength=0` the label is related to the features only
    through the site: countries differ in prevalence, and the site is legible
    from the recording. A model therefore cannot learn anything about TB, and
    should still post a respectable same-site score. That is precisely the
    result the field has been publishing.
    """
    X, meta = make_cohort(device_strength=5.0, disease_strength=0.0, seed=0)
    y = meta["label"].to_numpy()

    idx = np.arange(len(y))
    np.random.default_rng(0).shuffle(idx)
    cut = int(0.7 * len(idx))
    train, test = idx[:cut], idx[cut:]

    same_site = _person_level_auc(
        meta.iloc[test], _fit_predict(X[train], y[train], X[test])
    )
    null = prevalence_only_auc(y, meta["country"])

    assert same_site > 0.58, (
        f"same-site AUC was {same_site:.3f}. The cohort is built so that site "
        "prevalence alone is exploitable, so a chance-level score here means "
        "the shortcut is absent and the fixture is wrong."
    )
    # The tell: apparent skill is fully explained by the prevalence null.
    assert abs(same_site - null) < 0.10, (
        f"same-site AUC {same_site:.3f} should be close to the prevalence-only "
        f"null {null:.3f}, because there is nothing else in the data to learn."
    )


def test_leave_one_country_out_reduces_it_to_chance():
    """
    The same data and the same model, evaluated honestly.

    Held out an entire country, the site is unseen, its prevalence is unknown,
    and there is no disease signal to fall back on. The score must collapse to
    chance. If it does not, the protocol is not removing the shortcut and
    cannot be trusted on real audio.
    """
    X, meta = make_cohort(device_strength=5.0, disease_strength=0.0, seed=0)
    y = meta["label"].to_numpy()

    aucs = [
        score_fold(country, meta.iloc[te], _fit_predict(X[tr], y[tr], X[te])).auc
        for country, tr, te in leave_one_country_out(meta)
    ]
    mean_loco = float(np.nanmean(aucs))

    assert 0.42 < mean_loco < 0.58, (
        f"leave-one-country-out scored {mean_loco:.3f} on a cohort with no "
        "disease signal. Anything meaningfully above chance means the held-out "
        "site is still leaking."
    )


def test_no_participant_spans_a_fold(cohort):
    """The leak the protocol refuses to permit. Asserted, not assumed."""
    _, meta = cohort
    for country, tr, te in leave_one_country_out(meta):
        assert not (
            set(meta.iloc[tr]["participant_id"]) & set(meta.iloc[te]["participant_id"])
        ), f"participant leaked across the {country} fold"


def test_prevalence_alone_scores_above_chance(cohort):
    """
    The null model that has to be reported.

    Predicting each country's base rate, using no audio at all, beats chance
    because prevalence varies by site. Any real score near this one is a model
    that learned geography.
    """
    _, meta = cohort
    auc = prevalence_only_auc(meta["label"], meta["country"])
    assert auc > 0.55, (
        f"prevalence-only AUC was {auc:.3f}; if site base rates carried no "
        "signal there would be nothing for a model to exploit"
    )


def test_specificity_at_fixed_sensitivity_is_the_reported_number():
    """A perfect ranker gives perfect specificity; a coin flip gives none."""
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    perfect = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    spec, _ = specificity_at_sensitivity(y, perfect, 0.90)
    assert spec == pytest.approx(1.0)

    useless = np.full(8, 0.5)
    spec_flat, _ = specificity_at_sensitivity(y, useless, 0.90)
    assert spec_flat == pytest.approx(0.0)
