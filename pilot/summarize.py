"""Summarise pilot results: mean ± sd table, Qini-vs-size plot, Qini curves.

    python summarize.py --out results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LABELS = {"random": "Random", "lgbm_t": "LightGBM T", "lgbm_s": "LightGBM S",
          "lgbm_t_tuned": "LightGBM T (tuned)", "lgbm_s_tuned": "LightGBM S (tuned)",
          "tabpfn_t": "TabPFN T", "tabpfn_s": "TabPFN S", "causalpfn": "CausalPFN"}

# Paired comparisons: same split and same training rows, so differences are not split luck.
PAIRS = [("tabpfn_t", "lgbm_s_tuned"), ("tabpfn_t", "lgbm_t_tuned"), ("tabpfn_s", "lgbm_s_tuned"),
         ("causalpfn", "tabpfn_t"), ("lgbm_s_tuned", "lgbm_s")]


def paired(ok: pd.DataFrame) -> str:
    wide = ok.pivot_table(index=["size", "seed"], columns="method", values="qini")
    lines = ["\n### Paired comparisons (Qini, same splits)\n",
             "| Size | A | B | Mean A − B | SD | A wins | Splits |", "|---|---|---|---|---|---|---|"]
    for size, g in wide.groupby(level="size"):
        for a, b in PAIRS:
            if a in g and b in g:
                d = (g[a] - g[b]).dropna()
                if len(d):
                    lines.append(f"| {size} | {LABELS[a]} | {LABELS[b]} | {d.mean():+.4f} | "
                                 f"{d.std():.4f} | {(d > 0).sum()} | {len(d)} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    a = p.parse_args()
    out = Path(a.out)
    df = pd.read_csv(out / "results.csv")
    errors = df[df["error"].fillna("") != ""]
    if len(errors):
        print("Failed runs:\n", errors[["seed", "size", "method", "error"]].to_string(index=False))
    ok = df[df["error"].fillna("") == ""]

    agg = ok.groupby(["size", "method"]).agg(
        qini=("qini", "mean"), qini_sd=("qini", "std"), auuc=("auuc", "mean"),
        auuc_sd=("auuc", "std"), up10=("uplift_at_10", "mean"), sec=("seconds", "mean"),
        runs=("seed", "count")).reset_index()
    agg.to_csv(out / "summary.csv", index=False)
    with open(out / "summary.md", "w") as f:
        for size, g in agg.groupby("size"):
            f.write(f"\n### Training size {size}\n\n| Method | Qini | AUUC | Uplift@10% | Seconds | Runs |\n|---|---|---|---|---|---|\n")
            for _, r in g.sort_values("qini", ascending=False).iterrows():
                f.write(f"| {LABELS.get(r.method, r.method)} | {r.qini:.4f} ± {r.qini_sd:.4f} | "
                        f"{r.auuc:.4f} ± {r.auuc_sd:.4f} | {r.up10:.4f} | {r.sec:.1f} | {r.runs} |\n")
        f.write(paired(ok))
    print((out / "summary.md").read_text())

    # Qini vs training size
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, g in agg.groupby("method"):
        ax.errorbar(g["size"], g["qini"], yerr=g["qini_sd"], marker="o", capsize=3, label=LABELS.get(m, m))
    ax.set_xscale("log"); ax.set_xlabel("Training rows (context size)"); ax.set_ylabel("Qini AUC (test)")
    ax.axhline(0, color="grey", lw=0.8); ax.legend(frameon=False); ax.set_title("Hillstrom pilot: Qini by training size")
    fig.tight_layout(); fig.savefig(out / "qini_vs_size.png", dpi=200)

    # Qini curves at the largest size, seed 0
    from sklift.metrics import qini_curve

    T, Y = np.load(out / "T.npy"), np.load(out / "Y.npy")
    seed = int(ok["seed"].min()); size = int(ok["size"].max())
    test = np.load(out / f"test_idx_s{seed}.npy")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m in ok["method"].unique():
        f = out / "tau" / f"{m}_n{size}_s{seed}.npy"
        if f.exists():
            x, y = qini_curve(Y[test], np.load(f), T[test])
            ax.plot(x / x.max(), y, label=LABELS.get(m, m))
    ax.set_xlabel("Share of customers targeted"); ax.set_ylabel("Incremental visits (Qini)")
    ax.legend(frameon=False); ax.set_title(f"Qini curves, {size} training rows, seed {seed}")
    fig.tight_layout(); fig.savefig(out / "qini_curves.png", dpi=200)


if __name__ == "__main__":
    main()
