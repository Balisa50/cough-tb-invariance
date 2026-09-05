"""
Leave-one-site-out over both arms, saving each fold as it completes.

The first attempt at this experiment ran for hours and was lost twice: once to
the machine sleeping, once to the shell being torn down. Both times every
completed fold went with it, because results were only written after all
eighteen finished. Folds are deterministic under a fixed seed, verified by
reproducing all nine baseline folds to three decimals, so a fold already on
disk never needs recomputing.

    python scripts/run_resumable.py --root data/raw/coughvid/public_dataset

Re-running picks up wherever it stopped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import manifest_from_coughvid, manifest_summary  # noqa: E402
from src.evaluation import leave_one_country_out                # noqa: E402
from src.train import Config, run_fold                          # noqa: E402


def as_record(report, arm: str) -> dict:
    r = report.result
    return {
        "arm": arm,
        "site": r.held_out,
        "auc": r.auc,
        "spec_at_90_sens": r.specificity_at_90_sens,
        "prevalence": r.prevalence,
        "site_probe_accuracy": report.site_probe_accuracy,
        "site_chance": report.site_chance,
        "site_leak": report.site_leak,
        "prevalence_null_auc": report.prevalence_null_auc,
        "n_participants": r.n_participants,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="data/raw/coughvid/public_dataset")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/coughvid_folds.json")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out.exists():
        for rec in json.loads(out.read_text()):
            done[(rec["arm"], rec["site"])] = rec
        print(f"resuming: {len(done)} folds already on disk", flush=True)

    manifest = manifest_from_coughvid(args.root)
    print(manifest_summary(manifest), flush=True)

    arms = {
        "baseline": Config(max_lambda=0.0, augment=False, epochs=args.epochs,
                           seed=args.seed, crop="loudest"),
        "intervention": Config(max_lambda=1.0, augment=True, epochs=args.epochs,
                               seed=args.seed, crop="loudest"),
    }

    grouped = manifest.rename(columns={"site": "country"})
    folds = list(leave_one_country_out(grouped))

    for arm, cfg in arms.items():
        print(f"\n{arm.upper()}", flush=True)
        for site, train_idx, test_idx in folds:
            if (arm, site) in done:
                print(f"  {site:<6} cached", flush=True)
                continue
            t0 = time.time()
            report = run_fold(manifest, site, train_idx, test_idx, cfg)
            done[(arm, site)] = as_record(report, arm)
            # Written after every fold, so an interrupted run loses at most one.
            out.write_text(json.dumps(list(done.values()), indent=2))
            print(f"  {site:<6} AUC={report.result.auc:.3f}  "
                  f"spec@90={report.result.specificity_at_90_sens:.3f}  "
                  f"leak={report.site_leak:+.3f}  "
                  f"null={report.prevalence_null_auc:.3f}  "
                  f"[{time.time() - t0:.0f}s]", flush=True)

    print(f"\ncomplete: {len(done)} folds -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
