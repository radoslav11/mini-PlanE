#!/usr/bin/env python
# ruff: noqa: E402
"""Publish a saved Polaris-training checkpoint to the Polaris hub leaderboard.

Loads `experiments/train_polaris.py`'s best-val checkpoint, re-runs the model
on the benchmark's test split, evaluates server-side, and uploads to the
leaderboard. Submission metadata (name, description, GitHub URL, paper URL,
tags, user attributes) is configurable via CLI flags so this script can be
reused across benchmarks.

Authentication: you must `polaris login` first (or
`PolarisHubClient().login()`). Account at https://polarishub.io.
"""

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import polaris as po
import torch
from torch_geometric.loader import DataLoader

d_sage = str(ROOT / ".sage")
os.makedirs(d_sage, exist_ok=True)
os.environ.setdefault("DOT_SAGE", d_sage)
os.environ.setdefault("SAGE_CACHE_DIR", d_sage)

from plane import PlanE
from train_polaris import N_ATOM_TYPE, N_BOND_TYPE, load_or_preprocess


_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def slugify_name(s):
    """Polaris requires submission names match /^[a-zA-Z0-9_-]+$/."""
    out = _NAME_RE.sub("-", s).strip("-")
    return out or "unnamed"


def main():
    p = argparse.ArgumentParser(description="Submit a PlanE checkpoint to Polaris hub")
    p.add_argument("--benchmark",   type=str, default="tdcommons/caco2-wang")
    p.add_argument("--owner",       type=str, required=True,
                   help="Your Polaris hub username (the leaderboard entry owner).")
    p.add_argument("--name",        type=str, default="mini-PlanE / E-BasePlanE (single seed)")
    p.add_argument("--description", type=str,
                   default=("E-BasePlanE on SMILES -> SPQR tree (Sage) + PlanE layers. "
                            "Single seed (42), 500 epochs, ~12 min on a laptop CPU. "
                            "See mini-PlanE repo for full pipeline."))
    p.add_argument("--github-url",  type=str,
                   default="https://github.com/radoslav11/mini-PlanE")
    p.add_argument("--paper-url",   type=str,
                   default="https://arxiv.org/abs/2307.01180")
    p.add_argument("--tag",         action="append", default=[],
                   help="Repeatable: --tag plane --tag gnn --tag spqr")
    p.add_argument("--checkpoint",  type=str, default=None,
                   help="Defaults to .checkpoints/polaris_<slug>_best.pt")
    p.add_argument("--cache-dir",   type=str, default=str(ROOT / ".dataset" / "polaris"))
    p.add_argument("--n-batch",     type=int, default=64)
    p.add_argument("--device",      choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--dry-run",     action="store_true",
                   help="Run evaluate but skip the upload (for sanity-checking).")
    args = p.parse_args()

    slug = args.benchmark.replace("/", "_")
    p_ckpt = args.checkpoint or str(
        ROOT / ".checkpoints" / f"polaris_{slug}_best.pt"
    )
    if not os.path.exists(p_ckpt):
        sys.exit(f"checkpoint not found: {p_ckpt}\n"
                 f"run `experiments/train_polaris.py --benchmark {args.benchmark}` first")

    print(f"benchmark : {args.benchmark}")
    print(f"checkpoint: {p_ckpt}")

    # --- benchmark + cached test set
    bench = po.load_benchmark(args.benchmark)
    train, test = bench.get_train_test_split()
    train_items = [(smi, float(y)) for smi, y in train]
    test_items  = [(smi, 0.0) for smi in test.inputs]  # pyright: ignore[reportAttributeAccessIssue]
    _, te_ds = load_or_preprocess(
        args.benchmark, train_items, test_items,
        Path(args.cache_dir), n_workers=1,
    )
    print(f"  test {len(te_ds)} graphs")

    # --- model: shape from the checkpoint args, weights from state_dict
    ckpt = torch.load(p_ckpt, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    device = torch.device(args.device)
    d_edge = 0 if a.get("no_edge_feat", False) else N_BOND_TYPE
    model = PlanE(
        d_node=N_ATOM_TYPE, n_cls=1, d_edge=d_edge,
        d_hid=a["d_hid"], n_layers=a["n_layers"], d_pe=a["d_pe"],
        p_drop=a.get("p_drop", 0.0),
    ).to(device)
    # Materialize lazy modules, then load weights.
    with torch.no_grad():
        b = next(iter(DataLoader(te_ds, batch_size=min(args.n_batch, len(te_ds))))).to(device)
        model(b)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # --- regenerate test predictions
    preds = []
    loader = DataLoader(te_ds, batch_size=args.n_batch)
    with torch.no_grad():
        for b in loader:
            b = b.to(device)
            preds.append(model(b).view(-1).cpu())
    preds = torch.cat(preds).tolist()
    print(f"  predictions: {len(preds)}")

    # --- server-side score + metadata + upload
    results = bench.evaluate(preds)
    print("\nPolaris evaluate():")
    print(results)

    safe_name = slugify_name(args.name)
    if safe_name != args.name:
        print(f"  name slugified: {args.name!r} -> {safe_name!r} "
              f"(Polaris requires /^[a-zA-Z0-9_-]+$/)")
    results.name        = safe_name
    results.description = args.description
    results.tags        = args.tag or ["plane", "e-baseplane", "spqr", "gnn"]
    results.user_attributes = {
        "Framework":  "PyTorch / PyG",
        "Model":      "E-BasePlanE",
        "Backbone":   "SPQR + Block-Cut tree",
        "Source":     "mini-PlanE",
        "GitHub URL": args.github_url,
        "Paper URL":  args.paper_url,
    }

    if args.dry_run:
        print("\n--dry-run set, skipping upload_to_hub.")
        return

    print(f"\nUploading as owner={args.owner!r}...")
    results.upload_to_hub(owner=args.owner)
    print("done.")


if __name__ == "__main__":
    main()
