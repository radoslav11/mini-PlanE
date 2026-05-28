#!/usr/bin/env python
"""Plot the ZINC training curve from `zinc_log.csv` -> PNG."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / ".checkpoints" / "zinc_log.csv"
PNG_PATH = ROOT / ".checkpoints" / "zinc_log.png"

# Paper baselines from Table 6 (ZINC 12k subset).
REF_LINES = [
    (0.387, "GIN (no edges)",        "#bbbbbb"),
    (0.252, "GIN-E (with edges)",    "#999999"),
    (0.188, "PNA-E (with edges)",    "#777777"),
    (0.124, "BasePlanE (paper)",     "#ff7f0e"),
    (0.079, "CIN (with edges)",      "#9467bd"),
    (0.076, "E-BasePlanE (target)",  "#d62728"),
]


def main():
    epochs, lr, train, val, test, best = [], [], [], [], [], []
    with open(CSV_PATH) as fh:
        for row in csv.DictReader(fh):
            epochs.append(int(row["epoch"]))
            lr.append(float(row["lr"]))
            train.append(float(row["train_mae"]))
            val.append(float(row["val_mae"]))
            test.append(float(row["test_mae"]))
            best.append(float(row["best_val_mae"]))
    if not epochs:
        raise SystemExit(f"empty log at {CSV_PATH}")

    fig, (ax, ax_lr) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw={"height_ratios": [4, 1]},
    )

    ax.plot(epochs, train, label="train", color="#1f77b4", lw=1.5)
    ax.plot(epochs, val,   label="val",   color="#2ca02c", lw=1.5)
    ax.plot(epochs, test,  label="test",  color="#d62728", lw=1.5, alpha=0.8)
    ax.plot(epochs, best,  label="best val (so far)",
            color="#2ca02c", lw=2.0, ls=":", alpha=0.6)

    for mae, label, color in REF_LINES:
        ax.axhline(mae, color=color, lw=1.0, ls="--", alpha=0.6)
        ax.text(epochs[-1], mae, f"  {label}  ({mae:.3f})",
                color=color, fontsize=8, va="center")

    ax.set_yscale("log")
    ax.set_ylabel("MAE  (log scale)")
    ax.set_title(
        f"ZINC 12k — E-BasePlanE (mini-PlanE), "
        f"epoch {epochs[-1]}/{epochs[-1]} so far  "
        f"best val {min(val):.4f}  best test {min(test):.4f}"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", alpha=0.2)

    ax_lr.plot(epochs, lr, color="#8c564b", lw=1.2)
    ax_lr.set_yscale("log")
    ax_lr.set_xlabel("epoch")
    ax_lr.set_ylabel("lr")
    ax_lr.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=120)
    print(f"saved {PNG_PATH}  ({len(epochs)} epochs)")


if __name__ == "__main__":
    main()
