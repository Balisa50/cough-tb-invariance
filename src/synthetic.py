"""
A synthetic cohort that reproduces the confound, so the protocol can be
validated before the real data arrives.

Zhang et al. (arXiv 2608.25846) found three things in the real data:

  - representations cluster by recording device and dataset, not TB status
  - acquisition variability drives generalisation failure more than population
    shift does
  - predicted TB probability tracks country-level prevalence

This generator builds a cohort with exactly those properties and nothing else:
a strong per-country device signature, per-country prevalence that varies, and
a genuinely weak disease signal. Any evaluation worth using must show a model
scoring well under a random split and collapsing under leave-one-country-out.
If a protocol cannot detect the failure in data built to contain it, it will
not detect it in the wild.

The point is not realism. It is that the failure mode is present by
construction and the ground truth is known, which is never true of real audio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Participant counts from the real CODA cohort, so fold sizes and the
# imbalance between sites are representative rather than invented.
COUNTRY_N = {
    "Uganda": 487, "Philippines": 388, "Vietnam": 321, "South Africa": 275,
    "India": 241, "Madagascar": 234, "Tanzania": 197,
}

# TB prevalence differs markedly by site in CODA. This is what lets a model
# score well by identifying the country and predicting its base rate.
COUNTRY_PREVALENCE = {
    "Uganda": 0.32, "Philippines": 0.18, "Vietnam": 0.12, "South Africa": 0.41,
    "India": 0.25, "Madagascar": 0.15, "Tanzania": 0.28,
}


def make_cohort(
    n_features: int = 32,
    clips_per_person: int = 8,
    device_strength: float = 3.0,
    disease_strength: float = 0.45,
    seed: int = 42,
):
    """
    Build a synthetic cohort.

    `device_strength` is deliberately far larger than `disease_strength`. That
    ordering is the finding being reproduced: the equipment is louder in the
    features than the illness is.

    Returns (features, metadata). Metadata carries participant_id, country,
    label and device, so the protocol can group and hold out correctly.
    """
    rng = np.random.default_rng(seed)

    # Each site gets its own fixed direction in feature space, standing in for
    # a phone model's frequency response.
    device_signature = {
        c: rng.normal(0, 1, n_features) for c in COUNTRY_N
    }
    # One shared direction encodes the disease, identically at every site.
    # This is the only thing a model *should* be able to use.
    disease_direction = rng.normal(0, 1, n_features)

    # THE SHORTCUT.
    #
    # A device signature that is merely a constant per-country offset is not a
    # shortcut at all: it carries no information about who is ill, and a linear
    # model absorbs it and goes on to find the disease direction anyway. A
    # first version of this generator did exactly that, and leave-one-country-
    # out then scored *higher* than a random split, which is the opposite of
    # the published finding.
    #
    # The mechanism that actually breaks these models is entanglement. Sites
    # differ in prevalence, so within the training countries the device
    # signature is genuinely predictive of the label, and the model is right to
    # use it. Held out a new country, that same feature points somewhere else
    # and the prediction inverts. The shortcut is not noise; it is a real
    # correlation that does not transfer.
    #
    # This is encoded by tilting each device signature along the disease
    # direction in proportion to how far that site's prevalence sits from the
    # cohort mean.
    mean_prev = float(np.mean(list(COUNTRY_PREVALENCE.values())))
    for c in device_signature:
        tilt = (COUNTRY_PREVALENCE[c] - mean_prev) / mean_prev
        device_signature[c] = device_signature[c] + tilt * disease_direction

    rows, feats = [], []
    pid = 0
    for country, n_people in COUNTRY_N.items():
        prevalence = COUNTRY_PREVALENCE[country]
        for _ in range(n_people):
            pid += 1
            has_tb = int(rng.random() < prevalence)
            # A per-person offset: two clips from one larynx resemble each
            # other more than two clips from different people. This is what
            # makes a random clip-level split leak.
            person_offset = rng.normal(0, 0.6, n_features)

            for _ in range(clips_per_person):
                x = (
                    rng.normal(0, 1, n_features)
                    + device_strength * device_signature[country]
                    + person_offset
                    + disease_strength * has_tb * disease_direction
                )
                feats.append(x)
                rows.append({
                    "participant_id": f"P{pid:05d}",
                    "country": country,
                    "label": has_tb,
                    "device": f"phone_{country.replace(' ', '_').lower()}",
                })

    meta = pd.DataFrame(rows)
    X = np.vstack(feats).astype(np.float32)

    # Shuffle so nothing downstream can depend on ordering by country.
    order = rng.permutation(len(meta))
    return X[order], meta.iloc[order].reset_index(drop=True)
