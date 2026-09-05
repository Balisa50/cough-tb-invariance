"""
Turn a results json into the markdown that goes in the README.

Numbers in a readme drift from the numbers in the code, and this repository
has one job that depends on being trusted, so the tables are generated rather
than typed. Re-run this after any experiment and paste, or diff it against the
readme to see whether the published claims still match the last run.

    python scripts/report_results.py results/coughvid_results.json
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path


def table(rows: list[dict]) -> str:
    out = ["| held out | clips | prevalence | AUC | spec@90 | prevalence-only AUC | site leak |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in sorted(rows, key=lambda x: x["site"]):
        out.append(
            f"| {r['site']} | {r['n_participants']} | {r['prevalence']:.2f} | "
            f"{r['auc']:.3f} | {r['spec_at_90_sens']:.3f} | "
            f"{r['prevalence_null_auc']:.3f} | {r['site_leak']:+.3f} |"
        )
    return "\n".join(out)


def summary(name: str, rows: list[dict]) -> str:
    auc = [r["auc"] for r in rows]
    leak = [r["site_leak"] for r in rows]
    null = [r["prevalence_null_auc"] for r in rows]
    return (
        f"**{name}** - mean AUC {st.mean(auc):.3f}, worst {min(auc):.3f}, "
        f"best {max(auc):.3f}; mean prevalence-only AUC {st.mean(null):.3f}; "
        f"mean site leak {st.mean(leak):+.3f}"
    )


def main(path: str) -> int:
    payload = json.loads(Path(path).read_text())
    arms = payload["arms"]
    print(f"dataset: {payload['dataset']}   config: {payload['config']}\n")
    for name, rows in arms.items():
        print(summary(name, rows))
    print()
    for name, rows in arms.items():
        print(f"### {name}\n")
        print(table(rows))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1
                          else "results/coughvid_results.json"))
