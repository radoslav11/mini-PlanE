#!/usr/bin/env python
# ruff: noqa: E402 — imports below the sys.path.insert / Sage-env setup
"""Train (E-)BasePlanE on ZINC (paper Section 7.4).

ZINC 12k subset: regression of penalised logP on molecular graphs.

Paper baselines (Section 7.4 / Appendix D.4):
  BasePlanE  (no edge feats)  : MAE 0.124
  E-BasePlanE (with edges)    : MAE 0.076

The slow part is SPQR preprocessing — Sage runs once per molecule. We
parallelize over CPU cores via `multiprocessing` with the `fork` start method
so each worker inherits the already-imported Sage interpreter (no per-worker
~8s reinit). Results are cached to disk so subsequent runs skip preprocess.
"""

import argparse
import csv
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path

# Paths are anchored at the repo root so the script works from any cwd.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Sage cache routing — must happen before importing `plane` (which loads Sage).
d_sage = str(ROOT / ".sage")
os.makedirs(d_sage, exist_ok=True)
os.environ.setdefault("DOT_SAGE", d_sage)
os.environ.setdefault("SAGE_CACHE_DIR", d_sage)

from plane import PlanE, planar_preprocess  # noqa: E402


N_ATOM_TYPE = 28   # ZINC atom-type vocabulary size
N_BOND_TYPE = 4    # ZINC bond-type vocabulary size


# ---------------------------------------------------------------------------
# Preprocessing


def _preprocess_one(d):
    """Worker: SPQR-preprocess one molecule + one-hot featurize x, edge_attr.

    `planar_preprocess` uses `torch.unique` on x and edge_attr to derive the
    canonical labels for the KHC encoding, so they must be 1-D integer tensors
    going in. Afterwards we replace them with the one-hot float versions the
    model's Linear embeddings expect.
    """
    d.x = d.x.long().view(-1)
    d.edge_attr = d.edge_attr.long().view(-1)
    out = planar_preprocess(d)
    out.x = F.one_hot(d.x, N_ATOM_TYPE).float()
    out.edge_attr = F.one_hot(d.edge_attr, N_BOND_TYPE).float()
    return out


def _preprocess_split(raw, n_workers, label):
    if n_workers <= 1:
        return [_preprocess_one(d) for d in tqdm(raw, desc=label)]
    with mp.Pool(n_workers) as pool:
        return list(tqdm(
            pool.imap(_preprocess_one, raw, chunksize=8),
            total=len(raw), desc=label,
        ))


def load_or_preprocess(d_raw, d_cache, split, n_workers):
    p_cache = os.path.join(d_cache, f"{split}.pt")
    if os.path.exists(p_cache):
        return torch.load(p_cache, weights_only=False)
    raw = list(ZINC(d_raw, subset=True, split=split))
    print(f"[{split}] preprocessing {len(raw)} graphs ({n_workers} workers)...")
    out = _preprocess_split(raw, n_workers, split)
    os.makedirs(d_cache, exist_ok=True)
    torch.save(out, p_cache)
    return out


# ---------------------------------------------------------------------------
# Train / eval


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`. Trains if optimizer is given, else evaluates.
    Returns (loss, mae) where loss is the criterion mean and mae is the
    per-graph L1 distance to the target (same as loss if criterion is L1).
    """
    is_train = optimizer is not None
    model.train(is_train)
    n_loss = 0.0
    n_abs = 0.0
    n_count = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for b in loader:
            b = b.to(device)
            out = model(b).view(-1)
            tgt = b.y.view(-1).float()
            loss = criterion(out, tgt)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            n_loss  += loss.item() * b.num_graphs
            n_abs   += (out - tgt).abs().sum().item()
            n_count += b.num_graphs
    return n_loss / n_count, n_abs / n_count


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# CLI


def build_argparser():
    # Defaults follow paper Appendix D.4 for ZINC subset:
    #   3 layers, 128 hidden, 16 PE, 500 epochs, batch 256, lr 1e-3 + plateau.
    p = argparse.ArgumentParser(description="Train (E-)BasePlanE on ZINC 12k")
    p.add_argument("--n-epochs",     type=int,   default=500)
    p.add_argument("--n-batch",      type=int,   default=256)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--d-hid",        type=int,   default=128)
    p.add_argument("--n-layers",     type=int,   default=3)
    p.add_argument("--d-pe",         type=int,   default=16)
    p.add_argument("--p-drop",       type=float, default=0.0)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--n-workers",    type=int,   default=max(1, mp.cpu_count() - 2))
    p.add_argument("--no-edge-feat", action="store_true",
                   help="Disable edge features (= BasePlanE, paper MAE 0.124)")
    p.add_argument("--save-dir",     type=str,   default=str(ROOT / ".checkpoints"))
    p.add_argument("--data-dir",     type=str,   default=str(ROOT / ".dataset" / "ZINC_raw"))
    p.add_argument("--cache-dir",    type=str,   default=str(ROOT / ".dataset" / "ZINC"))
    p.add_argument("--log-csv",      type=str,
                   default=str(ROOT / ".checkpoints" / "zinc_log.csv"))
    p.add_argument("--preprocess-only", action="store_true",
                   help="Just run preprocessing (and exit before training)")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return p


def _pick_device(prefer):
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    # MPS currently crashes on empty placeholder tensors that arise when a
    # batch has e.g. zero BC-tree edges at some level — see PyTorch issue
    # tracker. CPU is the reliable default on macOS.
    return torch.device("cpu")


def main():
    args = build_argparser().parse_args()
    device = _pick_device(args.device)
    print(f"device: {device}")
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    p_ckpt = os.path.join(args.save_dir, "zinc_best.pt")

    print("=" * 70)
    print("Loading ZINC...")
    ds_tr = load_or_preprocess(args.data_dir, args.cache_dir, "train", args.n_workers)
    ds_va = load_or_preprocess(args.data_dir, args.cache_dir, "val",   args.n_workers)
    ds_te = load_or_preprocess(args.data_dir, args.cache_dir, "test",  args.n_workers)
    print(f"  train {len(ds_tr)}  val {len(ds_va)}  test {len(ds_te)}")

    if args.preprocess_only:
        print("preprocess-only set → exiting before training.")
        return

    loader_tr = DataLoader(ds_tr, batch_size=args.n_batch, shuffle=True)
    loader_va = DataLoader(ds_va, batch_size=args.n_batch)
    loader_te = DataLoader(ds_te, batch_size=args.n_batch)

    d_edge = 0 if args.no_edge_feat else N_BOND_TYPE
    model = PlanE(
        d_node=N_ATOM_TYPE, n_cls=1, d_edge=d_edge,
        d_hid=args.d_hid, n_layers=args.n_layers, d_pe=args.d_pe,
        p_drop=args.p_drop,
    ).to(device)

    # Materialize lazy modules; use an unshuffled loader to keep RNG untouched.
    with torch.no_grad():
        b_warmup = next(iter(
            DataLoader(ds_tr, batch_size=min(args.n_batch, len(ds_tr)))
        )).to(device)
        model(b_warmup)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params:,} params  d_edge={d_edge}  "
          f"({'E-BasePlanE' if d_edge > 0 else 'BasePlanE'})")

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    # Paper: halve LR every 25 epochs without validation improvement.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=25, min_lr=1e-6,
    )
    criterion = nn.L1Loss()

    # Per-epoch CSV log so the trajectory is plottable even if stdout is
    # buffered (e.g. when the script runs in the background).
    f_log = open(args.log_csv, "w", buffering=1)  # line-buffered
    writer = csv.writer(f_log)
    writer.writerow(["epoch", "lr", "train_mae", "val_mae", "test_mae",
                     "best_val_mae", "test_at_best_val"])

    print("\nTraining...")
    print("=" * 70)
    mae_best_va = float("inf")
    mae_at_best_te = float("inf")
    i_best = 0
    for i_ep in range(1, args.n_epochs + 1):
        _, mae_tr = run_epoch(model, loader_tr, criterion, device, optimizer=optimizer)
        _, mae_va = run_epoch(model, loader_va, criterion, device)
        _, mae_te = run_epoch(model, loader_te, criterion, device)
        scheduler.step(mae_va)

        if mae_va < mae_best_va:
            mae_best_va = mae_va
            mae_at_best_te = mae_te
            i_best = i_ep
            torch.save({
                "epoch": i_ep, "model_state": model.state_dict(),
                "best_val_mae": mae_best_va, "test_mae_at_best_val": mae_te,
                "args": vars(args),
            }, p_ckpt)

        lr_now = optimizer.param_groups[0]["lr"]
        writer.writerow([i_ep, lr_now, mae_tr, mae_va, mae_te,
                         mae_best_va, mae_at_best_te])

        if i_ep == 1 or i_ep % 5 == 0:
            print(
                f"ep {i_ep:03d}  tr {mae_tr:.4f}  va {mae_va:.4f}  te {mae_te:.4f}  "
                f"best_va {mae_best_va:.4f} (te {mae_at_best_te:.4f}) @ {i_best}  "
                f"lr {lr_now:.1e}"
            )

    f_log.close()
    print("=" * 70)
    print(f"best val MAE {mae_best_va:.4f}  test MAE {mae_at_best_te:.4f}  "
          f"@ epoch {i_best}  ({p_ckpt})")
    print(f"per-epoch log: {args.log_csv}")


if __name__ == "__main__":
    # Default `spawn` on macOS forces every worker to re-import Sage (~8s each).
    # `fork` inherits the parent's already-loaded interpreters, much faster.
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    main()
