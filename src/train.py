"""
Training, and the audit that decides whether the result means anything.

One fold is: train on every site but one, score the held-out site, then ask a
fresh probe whether the site is still recoverable from the encoder's features.
The probe is not decoration. Without it a good held-out score is ambiguous,
because a model can survive one held-out site by luck while still encoding the
site, and the next site is where it fails.

Both arms of the experiment run through this same function. The baseline is
lambda = 0 with augmentation off; the intervention turns one or both on. Using
one code path means a difference between arms cannot come from an accidental
difference in training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import CoughDataset
from .evaluation import FoldResult, prevalence_only_auc, score_fold
from .models import SiteInvariantClassifier, SiteProbe, lambda_schedule


@dataclass
class Config:
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 1e-3
    embedding_dim: int = 128
    max_lambda: float = 1.0        # 0 disables the adversary
    augment: bool = False
    probe_epochs: int = 40
    seed: int = 0
    device: str = "cpu"
    # "head" for the synthetic fixture, "loudest" for real recordings
    # where the cough is somewhere inside several seconds of audio.
    crop: str = "head"

    @property
    def arm(self) -> str:
        parts = []
        if self.max_lambda > 0:
            parts.append("adversarial")
        if self.augment:
            parts.append("augmented")
        return "+".join(parts) if parts else "baseline"


@dataclass
class FoldReport:
    result: FoldResult
    site_probe_accuracy: float
    site_chance: float
    prevalence_null_auc: float
    arm: str

    @property
    def site_leak(self) -> float:
        """
        How much site information survives, above chance.

        Zero means the encoder genuinely cannot tell the sites apart. This is
        the number that distinguishes invariance from a lucky fold.
        """
        return self.site_probe_accuracy - self.site_chance

    def __str__(self) -> str:
        return (
            f"  {self.result.held_out:<14} AUC={self.result.auc:.3f}  "
            f"spec@90={self.result.specificity_at_90_sens:.3f}  "
            f"site-leak={self.site_leak:+.3f}"
        )


def _epoch(model, loader, optimiser, cfg, step, total_steps):
    model.train()
    for spec, label, site, _ in loader:
        spec, label, site = spec.to(cfg.device), label.to(cfg.device), site.to(cfg.device)

        # The adversary is ramped in rather than switched on at full strength.
        # At full weight from step zero the encoder never learns anything worth
        # protecting and collapses to features uninformative about everything,
        # which is invariance of a useless kind.
        model.lambda_ = (
            lambda_schedule(step, total_steps, cfg.max_lambda)
            if cfg.max_lambda > 0 else 0.0
        )

        disease_logit, site_logits, _ = model(spec)
        loss = F.binary_cross_entropy_with_logits(disease_logit, label)
        if cfg.max_lambda > 0:
            # Gradient reversal sits inside the model, so adding this term
            # trains the site head normally and pushes the encoder the other way.
            loss = loss + F.cross_entropy(site_logits, site)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        step += 1
    return step


@torch.no_grad()
def _embed(model, loader, cfg):
    model.eval()
    scores, embeddings, sites, order = [], [], [], []
    for spec, _, site, idx in loader:
        logit, _, z = model(spec.to(cfg.device))
        scores.append(torch.sigmoid(logit).cpu().numpy())
        embeddings.append(z.cpu())
        sites.append(site)
        order.append(idx)
    return (
        np.concatenate(scores),
        torch.cat(embeddings),
        torch.cat(sites),
        torch.cat(order).numpy(),
    )


def _probe_site(embeddings, sites, n_sites, cfg) -> float:
    """
    Train a fresh probe on frozen features and report its accuracy.

    Deliberately trained to convergence on the very data it is tested on. The
    question is not whether site identity generalises; it is whether the
    information is present at all. Making the probe's job as easy as possible
    means a low score is strong evidence of absence rather than of a weak probe.
    """
    probe = SiteProbe(embeddings.shape[1], n_sites).to(cfg.device)
    optimiser = torch.optim.Adam(probe.parameters(), lr=1e-2)
    x, y = embeddings.detach().to(cfg.device), sites.to(cfg.device)

    for _ in range(cfg.probe_epochs):
        optimiser.zero_grad()
        F.cross_entropy(probe(x), y).backward()
        optimiser.step()

    with torch.no_grad():
        return float((probe(x).argmax(1) == y).float().mean().item())


def run_fold(manifest: pd.DataFrame, held_out, train_idx, test_idx, cfg: Config) -> FoldReport:
    torch.manual_seed(cfg.seed)

    train_meta = manifest.iloc[train_idx].reset_index(drop=True)
    test_meta = manifest.iloc[test_idx].reset_index(drop=True)

    # The site index covers training sites only. The held-out site is unseen by
    # construction, so it has no index and cannot be a target for the adversary.
    train_sites = sorted(train_meta["site"].unique())
    site_to_index = {s: i for i, s in enumerate(train_sites)}

    train_ds = CoughDataset(train_meta, site_to_index, augment=cfg.augment,
                            seed=cfg.seed, crop=cfg.crop)
    test_ds = CoughDataset(test_meta, {s: 0 for s in test_meta["site"].unique()},
                           augment=False, crop=cfg.crop)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    model = SiteInvariantClassifier(
        n_sites=len(train_sites), embedding_dim=cfg.embedding_dim,
        lambda_=cfg.max_lambda,
    ).to(cfg.device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    total_steps = max(cfg.epochs * len(train_loader), 1)
    step = 0
    for _ in range(cfg.epochs):
        step = _epoch(model, train_loader, optimiser, cfg, step, total_steps)

    scores, _, _, order = _embed(model, test_loader, cfg)
    result = score_fold(str(held_out), test_meta.iloc[order], scores)

    # The audit runs on training-site features: the question is whether the
    # encoder still separates the sites it was actually shown.
    _, train_embeddings, train_sites_tensor, _ = _embed(
        model, DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False), cfg
    )
    probe_accuracy = _probe_site(
        train_embeddings, train_sites_tensor, len(train_sites), cfg
    )

    return FoldReport(
        result=result,
        site_probe_accuracy=probe_accuracy,
        # Chance for the probe is the majority class, not 1/n, because sites
        # differ in size and predicting the largest is the trivial strategy.
        site_chance=float(
            train_meta["site"].value_counts(normalize=True).max()
        ),
        # Computed across the whole cohort, not within the held-out fold. A
        # single site has one prevalence, so the within-fold value is 0.5 by
        # construction and says nothing. The meaningful quantity is how much
        # was on offer to a model that could identify the site at all, which
        # is a property of the cohort.
        prevalence_null_auc=prevalence_only_auc(
            manifest["label"], manifest["site"]
        ),
        arm=cfg.arm,
    )


def run_experiment(manifest: pd.DataFrame, cfg: Config, verbose: bool = True) -> list[FoldReport]:
    """Leave-one-site-out across the whole manifest."""
    from .evaluation import leave_one_country_out

    grouped = manifest.rename(columns={"site": "country"})
    reports = []
    for site, train_idx, test_idx in leave_one_country_out(grouped):
        report = run_fold(manifest, site, train_idx, test_idx, cfg)
        reports.append(report)
        if verbose:
            print(str(report), flush=True)
    return reports


def compare(baseline: list[FoldReport], treated: list[FoldReport]) -> str:
    """
    The comparison the paper turns on.

    Leads with the worst site, because a screening tool is limited by where it
    performs worst, and reports the change in site leak alongside, because an
    accuracy gain without a fall in leak is not invariance.
    """
    def stats(reports):
        aucs = np.array([r.result.auc for r in reports if r.result.auc == r.result.auc])
        specs = np.array([
            r.result.specificity_at_90_sens for r in reports
            if r.result.specificity_at_90_sens == r.result.specificity_at_90_sens
        ])
        leaks = np.array([r.site_leak for r in reports])
        return aucs, specs, leaks

    b_auc, b_spec, b_leak = stats(baseline)
    t_auc, t_spec, t_leak = stats(treated)

    return "\n".join([
        "",
        "BASELINE vs INTERVENTION",
        "=" * 62,
        f"  {'':22} {'baseline':>12} {'treated':>12} {'change':>10}",
        f"  {'mean AUC':22} {b_auc.mean():>12.3f} {t_auc.mean():>12.3f} "
        f"{t_auc.mean() - b_auc.mean():>+10.3f}",
        f"  {'worst-site AUC':22} {b_auc.min():>12.3f} {t_auc.min():>12.3f} "
        f"{t_auc.min() - b_auc.min():>+10.3f}",
        f"  {'worst-site spec@90':22} {b_spec.min():>12.3f} {t_spec.min():>12.3f} "
        f"{t_spec.min() - b_spec.min():>+10.3f}",
        f"  {'site leak (mean)':22} {b_leak.mean():>12.3f} {t_leak.mean():>12.3f} "
        f"{t_leak.mean() - b_leak.mean():>+10.3f}",
        "=" * 62,
        f"  cohort prevalence-only null: {baseline[0].prevalence_null_auc:.3f}"
        if baseline else "",
        "  A rise in worst-site specificity with no fall in site leak is not",
        "  invariance. It is a better model that still knows where it is.",
    ])
