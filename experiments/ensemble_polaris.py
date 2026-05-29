#!/usr/bin/env python
# ruff: noqa: E402
"""Ensemble several PlanE checkpoints on a Polaris benchmark.

Loads N checkpoints (each saved by train_polaris.py), regenerates each one's
test predictions (respecting that checkpoint's own --rich-features /
--normalize-target settings, read from its saved args), averages the
predictions, and scores via bench.evaluate(). Optionally uploads.

Target normalization stats are recomputed from the benchmark's train targets
(deterministic), since train_polaris.py doesn't store y_mean/y_std.

Usage:
    python experiments/ensemble_polaris.py \
        --ckpt .checkpoints/runC/polaris_tdcommons_caco2-wang_best.pt \
        --ckpt .checkpoints/runD/polaris_tdcommons_caco2-wang_best.pt
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
import polaris as po
import torch
from torch_geometric.loader import DataLoader

d_sage = str(ROOT / ".sage")
os.makedirs(d_sage, exist_ok=True)
os.environ.setdefault("DOT_SAGE", d_sage)
os.environ.setdefault("SAGE_CACHE_DIR", d_sage)

from plane import PlanE
from train_polaris import (
    N_ATOM_TYPE,
    N_BOND_TYPE,
    N_RICH_ATOM_FEATS,
    N_RICH_BOND_FEATS,
    load_or_preprocess,
)
from submit_polaris import slugify_name


def predict_one(ckpt_path, benchmark, train_items, test_items, cache_dir,
                y_mean, y_std, device):
    """Return this checkpoint's (un-normalized) test predictions as a np array."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    rich = a.get("rich_features", False)
    norm = a.get("normalize_target", False)

    _, te_ds = load_or_preprocess(
        benchmark, train_items, test_items, Path(cache_dir),
        n_workers=1, rich=rich,
    )

    d_node_in = N_RICH_ATOM_FEATS if rich else N_ATOM_TYPE
    d_edge_in = (0 if a.get("no_edge_feat", False)
                 else (N_RICH_BOND_FEATS if rich else N_BOND_TYPE))
    model = PlanE(
        d_node=d_node_in, n_cls=1, d_edge=d_edge_in,
        d_hid=a["d_hid"], n_layers=a["n_layers"], d_pe=a["d_pe"],
        p_drop=a.get("p_drop", 0.0),
    ).to(device)
    loader = DataLoader(te_ds, batch_size=a.get("n_batch", 64))
    with torch.no_grad():
        model(next(iter(loader)).to(device))   # materialize lazy modules
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds = []
    with torch.no_grad():
        for b in loader:
            preds.append(model(b.to(device)).view(-1).cpu())
    preds = torch.cat(preds).numpy()
    if norm:
        preds = preds * y_std + y_mean
    return preds


def main():
    p = argparse.ArgumentParser(description="Ensemble PlanE checkpoints on Polaris")
    p.add_argument("--benchmark", type=str, default="tdcommons/caco2-wang")
    p.add_argument("--ckpt", action="append", required=True,
                   help="Repeatable: --ckpt path1 --ckpt path2 ...")
    p.add_argument("--cache-dir", type=str, default=str(ROOT / ".dataset" / "polaris"))
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--owner", type=str, default=None,
                   help="If set, upload the ensemble result under this owner.")
    p.add_argument("--name", type=str, default="mini-PlanE-ensemble")
    p.add_argument("--description", type=str,
                   default="Ensemble of E-BasePlanE checkpoints (rich features + "
                           "z-scored target). Mean of per-model test predictions.")
    args = p.parse_args()

    device = torch.device(args.device)
    bench = po.load_benchmark(args.benchmark)
    train, test = bench.get_train_test_split()
    train_items = [(smi, float(y)) for smi, y in train]
    test_items  = [(smi, 0.0) for smi in test.inputs]  # pyright: ignore[reportAttributeAccessIssue]

    # Deterministic z-score stats from the train targets.
    ys = np.array([y for _, y in train_items], dtype=np.float64)
    y_mean, y_std = float(ys.mean()), float(ys.std()) + 1e-9
    print(f"benchmark {args.benchmark}  train {len(train_items)}  "
          f"test {len(test_items)}  y_mean={y_mean:.4f} y_std={y_std:.4f}")

    all_preds = []
    for c in args.ckpt:
        if not os.path.exists(c):
            print(f"  skip (missing): {c}")
            continue
        preds = predict_one(c, args.benchmark, train_items, test_items,
                            args.cache_dir, y_mean, y_std, device)
        all_preds.append(preds)
        single = bench.evaluate(preds.tolist())
        print(f"  {c}\n    single -> {single.results.to_dict('records')}")

    if not all_preds:
        sys.exit("no usable checkpoints")

    avg = np.mean(np.stack(all_preds, axis=0), axis=0)
    results = bench.evaluate(avg.tolist())
    print(f"\nENSEMBLE of {len(all_preds)} models:")
    print(results.results.to_dict("records"))

    if args.owner:
        results.name = slugify_name(args.name)
        results.description = args.description
        results.tags = ["plane", "e-baseplane", "ensemble", "spqr", "gnn"]
        results.user_attributes = {
            "Model": "E-BasePlanE ensemble", "Source": "mini-PlanE",
            "Members": str(len(all_preds)),
        }
        print(f"\nuploading as owner={args.owner!r}...")
        results.upload_to_hub(owner=args.owner)
        print("done.")


if __name__ == "__main__":
    main()
