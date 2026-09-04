#!/usr/bin/env python
"""
Run the whole thing.

    python run_experiment.py --dataset synthetic          # works today
    python run_experiment.py --dataset coswara --root ... --meta ...
    python run_experiment.py --dataset coda    --root ... --meta ...

Runs the baseline and the intervention over the same folds and prints the
comparison. Nothing here is dataset-specific: swapping corpora changes one
manifest function and nothing else.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data import (  # noqa: E402
    manifest_from_coda, manifest_from_coswara, manifest_from_synthetic,
    manifest_summary,
)
from src.train import Config, compare, run_experiment  # noqa: E402


def build_manifest(args) -> pd.DataFrame:
    if args.dataset == "synthetic":
        return manifest_from_synthetic(
            args.root or "data/interim/synthetic",
            n_per_site=args.n_per_site, n_sites=args.n_sites, seed=args.seed,
        )
    if args.dataset == "coswara":
        return manifest_from_coswara(args.root, args.meta)
    if args.dataset == "coda":
        return manifest_from_coda(args.root, args.meta)
    raise ValueError(f"unknown dataset: {args.dataset}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="synthetic",
                   choices=["synthetic", "coswara", "coda"])
    p.add_argument("--root", help="audio directory")
    p.add_argument("--meta", help="metadata csv")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-sites", type=int, default=4, help="synthetic only")
    p.add_argument("--n-per-site", type=int, default=40, help="synthetic only")
    p.add_argument("--out", default="results", help="where to write the json")
    args = p.parse_args()

    manifest = build_manifest(args)
    print(manifest_summary(manifest))

    if args.dataset == "synthetic":
        print(
            "\n  NOTE: synthetic audio. These numbers describe the pipeline, "
            "not\n  cough classification, and must never be reported as a result."
        )

    shared = dict(epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)

    print("\nBASELINE  (no adversary, no augmentation)")
    print("-" * 62)
    baseline = run_experiment(manifest, Config(max_lambda=0.0, augment=False, **shared))

    print("\nINTERVENTION  (adversarial site head + device simulation)")
    print("-" * 62)
    treated = run_experiment(manifest, Config(max_lambda=1.0, augment=True, **shared))

    print(compare(baseline, treated))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "config": shared,
        "arms": {
            arm: [
                {
                    "site": r.result.held_out,
                    "auc": r.result.auc,
                    "spec_at_90_sens": r.result.specificity_at_90_sens,
                    "prevalence": r.result.prevalence,
                    "site_probe_accuracy": r.site_probe_accuracy,
                    "site_chance": r.site_chance,
                    "site_leak": r.site_leak,
                    "prevalence_null_auc": r.prevalence_null_auc,
                    "n_participants": r.result.n_participants,
                }
                for r in reports
            ]
            for arm, reports in (("baseline", baseline), ("intervention", treated))
        },
    }
    path = out_dir / f"{args.dataset}_results.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\n  written: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
