"""
Does the machinery do what it claims?

These tests do not ask whether the model detects disease. They ask whether the
two mechanisms behave as advertised: that reversing the gradient actually
opposes the encoder, and that device simulation actually destroys the device
cue. Both are claims about the code, and both are checkable without real audio.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.audio import (  # noqa: E402
    SAMPLE_RATE, DeviceSimulator, fix_length, log_mel, synthetic_cough,
)
from src.models import (  # noqa: E402
    SiteInvariantClassifier, SiteProbe, lambda_schedule, reverse_gradient,
)


# ── the gradient reversal actually reverses ─────────────────────────────────

def test_gradient_reversal_is_identity_forwards():
    x = torch.randn(4, 8)
    assert torch.allclose(reverse_gradient(x, 1.0), x)


def test_gradient_reversal_flips_the_sign_backwards():
    """The whole mechanism in one assertion."""
    x = torch.randn(4, 8, requires_grad=True)
    reverse_gradient(x, 1.0).sum().backward()
    reversed_grad = x.grad.clone()

    x.grad = None
    x.sum().backward()
    plain_grad = x.grad.clone()

    assert torch.allclose(reversed_grad, -plain_grad), (
        "the reversal layer must negate the gradient reaching the encoder; "
        "without that the site head is just an extra classifier and nothing "
        "is being removed"
    )


def test_lambda_scales_the_opposition():
    x = torch.randn(4, 8, requires_grad=True)
    reverse_gradient(x, 0.5).sum().backward()
    half = x.grad.clone()
    x.grad = None
    reverse_gradient(x, 1.0).sum().backward()
    full = x.grad.clone()
    assert torch.allclose(half * 2, full)


def test_lambda_schedule_ramps_from_zero():
    assert lambda_schedule(0, 100) == pytest.approx(0.0, abs=1e-6)
    assert lambda_schedule(100, 100) > 0.9
    steps = [lambda_schedule(i, 100) for i in range(0, 101, 10)]
    assert all(b >= a for a, b in zip(steps, steps[1:])), "schedule must not decrease"


# ── the model runs and both heads are wired ─────────────────────────────────

def test_forward_pass_shapes():
    model = SiteInvariantClassifier(n_sites=7)
    spec = torch.randn(3, 64, 87)                      # a 0.5s clip at 44.1kHz
    disease, site, z = model(spec)
    assert disease.shape == (3,)
    assert site.shape == (3, 7)
    assert z.shape == (3, model.encoder.embedding_dim)


def test_adversary_reaches_the_encoder():
    """
    Site loss alone must produce gradient in the encoder.

    If it does not, the adversarial branch is disconnected and the method is
    silently doing nothing, which would still train and still report numbers.
    """
    model = SiteInvariantClassifier(n_sites=7, lambda_=1.0)
    _, site_logits, _ = model(torch.randn(4, 64, 87))
    torch.nn.functional.cross_entropy(
        site_logits, torch.randint(0, 7, (4,))
    ).backward()

    grads = [
        p.grad.abs().sum().item()
        for p in model.encoder.parameters() if p.grad is not None
    ]
    assert grads and sum(grads) > 0, "site loss never reached the encoder"


def test_lambda_zero_is_the_plain_baseline():
    """Both arms share one class, so a difference cannot be architectural."""
    model = SiteInvariantClassifier(n_sites=7, lambda_=0.0)
    _, site_logits, _ = model(torch.randn(4, 64, 87))
    torch.nn.functional.cross_entropy(
        site_logits, torch.randint(0, 7, (4,))
    ).backward()

    total = sum(
        p.grad.abs().sum().item()
        for p in model.encoder.parameters() if p.grad is not None
    )
    assert total == pytest.approx(0.0, abs=1e-8), (
        "with lambda=0 the adversary must not perturb the encoder at all"
    )


# ── device simulation destroys the device cue ───────────────────────────────

def test_synthetic_cough_carries_a_detectable_device_signature():
    """The fixture must contain the confound, or the next test proves nothing."""
    rng = np.random.default_rng(0)
    specs = {
        device: np.mean([
            log_mel(fix_length(synthetic_cough(False, device, rng)))
            for _ in range(6)
        ], axis=0)
        for device in (0, 3)
    }
    separation = np.abs(specs[0] - specs[3]).mean()
    assert separation > 0.05, (
        f"device signatures differ by only {separation:.4f}; the fixture does "
        "not contain the shortcut it is supposed to contain"
    )


def test_device_simulation_makes_the_device_harder_to_identify():
    """
    The intervention, measured the only way that means anything.

    An earlier version of this test compared averaged spectrograms and asked
    whether the distance between two devices shrank. That metric was wrong: the
    simulator deliberately adds randomness, so it inflates spectral distance
    even as it destroys the cue, and the test failed while the augmentation was
    working. Distance between spectrograms is not the quantity of interest.

    The quantity of interest is whether a classifier can still tell the devices
    apart. That is what the shortcut consists of, and it is what a site probe
    measures on real data, so the test measures it here too.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    rng = np.random.default_rng(0)
    sim = DeviceSimulator(seed=0)
    devices = (0, 3)

    def dataset(augment):
        X, y = [], []
        for device in devices:
            for _ in range(40):
                w = fix_length(synthetic_cough(False, device, rng))
                X.append(log_mel(sim(w) if augment else w).ravel())
                y.append(device)
        return np.array(X), np.array(y)

    def device_accuracy(augment):
        X, y = dataset(augment)
        return float(cross_val_score(
            LogisticRegression(max_iter=1000), X, y, cv=4, scoring="accuracy"
        ).mean())

    raw = device_accuracy(augment=False)
    augmented = device_accuracy(augment=True)

    assert raw > 0.85, (
        f"device was only {raw:.2f} identifiable in raw audio; the fixture is "
        "supposed to contain a strong device signature"
    )
    assert augmented < raw, (
        f"device identifiability did not fall under simulation "
        f"({raw:.2f} -> {augmented:.2f}); the augmentation is not attacking "
        "the cue it exists to attack"
    )


def test_simulation_is_random_per_call():
    """A fixed transform would just be a new, equally learnable device."""
    rng = np.random.default_rng(0)
    sim = DeviceSimulator(seed=0)
    w = fix_length(synthetic_cough(True, 1, rng))
    assert not np.allclose(sim(w), sim(w)), (
        "two calls produced identical audio; the simulator must sample a new "
        "device each time or it adds a constant the model can simply learn"
    )


def test_simulation_preserves_clip_shape_and_finiteness():
    rng = np.random.default_rng(0)
    sim = DeviceSimulator(seed=1)
    w = fix_length(synthetic_cough(True, 2, rng))
    out = sim(w)
    assert out.shape == w.shape
    assert np.isfinite(out).all(), "augmentation produced NaN or inf"


def test_probe_recovers_site_from_unprotected_features():
    """
    The audit instrument itself must work.

    Given features that plainly encode the site, the probe has to find it.
    A probe that fails here would report false invariance later.
    """
    torch.manual_seed(0)
    n_sites, dim = 4, 16
    z, y = [], []
    for site in range(n_sites):
        centre = torch.zeros(dim)
        centre[site] = 6.0                     # site is written into the features
        z.append(centre + 0.25 * torch.randn(64, dim))
        y.append(torch.full((64,), site))
    z, y = torch.cat(z), torch.cat(y)

    probe = SiteProbe(dim, n_sites)
    opt = torch.optim.Adam(probe.parameters(), lr=0.01)
    for _ in range(200):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(probe(z), y)
        loss.backward()
        opt.step()

    accuracy = (probe(z).argmax(1) == y).float().mean().item()
    assert accuracy > 0.9, (
        f"probe reached only {accuracy:.2f} on features that explicitly encode "
        "the site; it cannot be trusted to detect residual site information"
    )
