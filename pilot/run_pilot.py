"""Hillstrom pilot: every method x training size x seed, scored on a fixed test split per seed.

Resumable: rows already in <out>/results.csv without an error are skipped, so a rerun
(after a time limit or a crash) only does what is missing.

Example:
    python pilot/run_pilot.py --sizes 2000 8000 25000 --seeds 0 1 2 --out results/pilot
    python pilot/run_pilot.py --synthetic --methods random lgbm_t lgbm_s --device cpu --out results/smoke
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from data import load_hillstrom, make_split, subsample, synthetic_hillstrom
from methods import REGISTRY


def score(y, tau, T):
    from sklift.metrics import qini_auc_score, uplift_at_k, uplift_auc_score

    return {
        "qini": qini_auc_score(y, tau, T),
        "auuc": uplift_auc_score(y, tau, T),
        "uplift_at_10": uplift_at_k(y, tau, T, strategy="overall", k=0.1),
        "uplift_at_30": uplift_at_k(y, tau, T, strategy="overall", k=0.3),
    }


def versions():
    out = {"python": platform.python_version()}
    for pkg in ["tabpfn", "causalpfn", "scikit-uplift", "lightgbm", "scikit-learn", "torch"]:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[2000, 8000, 25000])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--methods", nargs="+", default=list(REGISTRY))
    p.add_argument("--device", default="cuda")
    p.add_argument("--causalpfn-ctx", type=int, default=4096, help="CausalPFN max_context_length")
    p.add_argument("--data-home", default=os.environ.get("UPLIFT_DATA"),
                   help="where scikit-uplift stores Hillstrom (default: $UPLIFT_DATA)")
    p.add_argument("--synthetic", action="store_true", help="use the offline stand-in data")
    p.add_argument("--out", default="results/pilot")
    a = p.parse_args()

    out = Path(a.out)
    (out / "tau").mkdir(parents=True, exist_ok=True)
    if a.synthetic:
        X, T, Y, _ = synthetic_hillstrom()
    else:
        X, T, Y = load_hillstrom(a.data_home)
    cols, X = list(X.columns), X.to_numpy(dtype=float)
    meta = {"args": vars(a), "versions": versions(), "n_rows": len(Y), "features": cols,
            "treated_share": float(T.mean()), "visit_rate": float(Y.mean())}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ["n_rows", "treated_share", "visit_rate"]}))

    np.save(out / "T.npy", T)
    np.save(out / "Y.npy", Y)
    csv = out / "results.csv"
    rows = []
    if csv.exists():
        prev = pd.read_csv(csv)
        rows = prev[prev["error"].fillna("") == ""].to_dict("records")
    done = {(r["seed"], r["size"], r["method"]) for r in rows}
    print(f"{len(done)} finished runs found; skipping them", flush=True)

    for seed in a.seeds:
        pool, test = make_split(len(Y), T, Y, seed)
        np.save(out / f"test_idx_s{seed}.npy", test)
        for size in a.sizes:
            tr = subsample(pool, T, Y, size, seed)
            for name in a.methods:
                if (seed, len(tr), name) in done:
                    continue
                t0 = time.time()
                row = {"seed": seed, "size": len(tr), "method": name}
                try:
                    tau = REGISTRY[name](Y[tr], T[tr], X[tr], X[test], seed=seed,
                                         device=a.device, max_ctx=a.causalpfn_ctx)
                    row.update(score(Y[test], tau, T[test]), seconds=time.time() - t0, error="")
                    np.save(out / "tau" / f"{name}_n{len(tr)}_s{seed}.npy", tau)
                except Exception as e:  # keep going; one failing method should not stop the run
                    row.update(seconds=time.time() - t0, error=f"{type(e).__name__}: {e}")
                rows.append(row)
                pd.DataFrame(rows).to_csv(csv, index=False)
                print(row, flush=True)


if __name__ == "__main__":
    main()
