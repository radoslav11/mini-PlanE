#!/usr/bin/env python
# ruff: noqa: E402 — imports below the sys.path.insert / Sage-env setup
"""Train (E-)BasePlanE on a Polaris benchmark.

Default task is `tdcommons/caco2-wang` (Caco-2 permeability regression,
728 train / 182 test, MAE metric). Switch via `--benchmark <owner/slug>`.

The pipeline is:
    SMILES  ->  rdkit Mol  ->  PyG `Data` (atom/bond one-hot)
            ->  `planar_preprocess` (SPQR + BC tree)  ->  PlanE training

Most drug-like organic molecules are planar (no K5/K3,3 minor), so
`planar_preprocess` succeeds; the few that don't are skipped and a count is
reported. Preprocessing runs in parallel via `multiprocessing.Pool(fork)`.
"""

import argparse
import csv
import multiprocessing as mp
import os
import pickle
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import polaris as po
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch import nn, optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

d_sage = str(ROOT / ".sage")
os.makedirs(d_sage, exist_ok=True)
os.environ.setdefault("DOT_SAGE", d_sage)
os.environ.setdefault("SAGE_CACHE_DIR", d_sage)

from plane import PlanE, planar_preprocess


RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue]


# ---------------------------------------------------------------------------
# SMILES -> PyG  featurization

# Drug-like atom set (atomic numbers). Anything else falls into the last bucket.
ATOM_LIST = [6, 7, 8, 9, 15, 16, 17, 35, 53, 5, 14, 1]  # C N O F P S Cl Br I B Si H
ATOM_TO_IDX = {a: i for i, a in enumerate(ATOM_LIST)}
N_ATOM_TYPE = len(ATOM_LIST) + 1   # +1 "other"

BOND_LIST = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]
BOND_TO_IDX = {b: i for i, b in enumerate(BOND_LIST)}
N_BOND_TYPE = len(BOND_LIST) + 1   # +1 "other"


def _atom_idx(atomic_num):
    return ATOM_TO_IDX.get(atomic_num, N_ATOM_TYPE - 1)


def _bond_idx(bond_type):
    return BOND_TO_IDX.get(bond_type, N_BOND_TYPE - 1)


def smiles_to_pyg(smi, y):
    """SMILES -> PyG Data with int atom/bond labels (1-D long tensors).

    `planar_preprocess` uses torch.unique on x and edge_attr to derive
    canonical labels for the KHC encoding, so they must be 1-D ints here.
    We later replace them with one-hot floats post-preprocess.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor(
        [_atom_idx(a.GetAtomicNum()) for a in mol.GetAtoms()], dtype=torch.long
    )

    src, dst, eattr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        t = _bond_idx(bond.GetBondType())
        src += [i, j]
        dst += [j, i]
        eattr += [t, t]
    if not src:
        return None   # isolated atoms, no bonds
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(eattr, dtype=torch.long)
    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        y=torch.tensor([y], dtype=torch.float),
    )


# ---------------------------------------------------------------------------
# Parallel preprocessing


def _preprocess_one(args):
    smi, y = args
    d = smiles_to_pyg(smi, y)
    if d is None:
        return None
    try:
        out = planar_preprocess(d)
    except Exception:
        return None   # non-planar / Sage edge-case
    out.x = F.one_hot(d.x, N_ATOM_TYPE).float()
    out.edge_attr = F.one_hot(d.edge_attr, N_BOND_TYPE).float()
    return out


def preprocess_split(items, n_workers, label):
    """`items` is a list of (smiles, target) tuples."""
    if n_workers <= 1:
        out = [_preprocess_one(it) for it in tqdm(items, desc=label)]
    else:
        with mp.Pool(n_workers) as pool:
            out = list(tqdm(
                pool.imap(_preprocess_one, items, chunksize=8),
                total=len(items), desc=label,
            ))
    n_dropped = sum(1 for x in out if x is None)
    if n_dropped:
        print(f"  [{label}] dropped {n_dropped}/{len(items)} "
              f"(non-planar or unparseable)")
    return [x for x in out if x is not None]


def load_or_preprocess(slug, train_items, test_items, cache_root, n_workers):
    d_cache = cache_root / slug.replace("/", "_")
    p_tr = d_cache / "train.pkl"
    p_te = d_cache / "test.pkl"
    if p_tr.exists() and p_te.exists():
        with open(p_tr, "rb") as fh:
            tr = pickle.load(fh)
        with open(p_te, "rb") as fh:
            te = pickle.load(fh)
        return tr, te
    d_cache.mkdir(parents=True, exist_ok=True)
    print(f"preprocessing [{slug}] with {n_workers} workers...")
    tr = preprocess_split(train_items, n_workers, "train")
    te = preprocess_split(test_items, n_workers, "test")
    with open(p_tr, "wb") as fh:
        pickle.dump(tr, fh)
    with open(p_te, "wb") as fh:
        pickle.dump(te, fh)
    return tr, te


# ---------------------------------------------------------------------------
# Train / eval


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    n_loss = n_abs = n_count = 0.0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    preds, tgts = [], []
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
            preds.append(out.detach().cpu())
            tgts.append(tgt.detach().cpu())
    return n_loss / n_count, n_abs / n_count, torch.cat(preds), torch.cat(tgts)


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
    p = argparse.ArgumentParser(description="Train (E-)BasePlanE on a Polaris benchmark")
    p.add_argument("--benchmark",    type=str,   default="tdcommons/caco2-wang")
    p.add_argument("--n-epochs",     type=int,   default=500)
    p.add_argument("--n-batch",      type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--d-hid",        type=int,   default=128)
    p.add_argument("--n-layers",     type=int,   default=3)
    p.add_argument("--d-pe",         type=int,   default=16)
    p.add_argument("--p-drop",       type=float, default=0.1)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--n-workers",    type=int,   default=max(1, mp.cpu_count() - 2))
    p.add_argument("--no-edge-feat", action="store_true",
                   help="Disable edge features (BasePlanE rather than E-BasePlanE)")
    p.add_argument("--save-dir",     type=str,   default=str(ROOT / ".checkpoints"))
    p.add_argument("--cache-dir",    type=str,   default=str(ROOT / ".dataset" / "polaris"))
    p.add_argument("--device",       choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--preprocess-only", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    print(f"benchmark: {args.benchmark}  device: {device}")

    # --- load benchmark
    # Polaris masks the test targets on purpose — we submit predictions back
    # via `bench.evaluate(preds)` and it grades server-side. So the train
    # split yields (smi, y) tuples while the test split yields bare SMILES.
    bench = po.load_benchmark(args.benchmark)
    train, test = bench.get_train_test_split()
    train_items = [(smi, float(y)) for smi, y in train]
    # `test.inputs` exists on single-test-set benchmarks; pyright's stub types
    # the split return as a dict in the multi-test-set case, hence the ignore.
    test_items  = [(smi, 0.0) for smi in test.inputs]   # pyright: ignore[reportAttributeAccessIssue]
    print(f"  train {len(train_items)}  test {len(test_items)}")

    # --- preprocess (cached)
    tr_ds, te_ds = load_or_preprocess(
        args.benchmark, train_items, test_items,
        Path(args.cache_dir), args.n_workers,
    )
    print(f"  after preprocess: train {len(tr_ds)}  test {len(te_ds)}")

    if args.preprocess_only:
        print("preprocess-only set -> exiting before training.")
        return

    loader_te = DataLoader(te_ds, batch_size=args.n_batch)

    # --- model
    d_edge = 0 if args.no_edge_feat else N_BOND_TYPE
    model = PlanE(
        d_node=N_ATOM_TYPE, n_cls=1, d_edge=d_edge,
        d_hid=args.d_hid, n_layers=args.n_layers, d_pe=args.d_pe,
        p_drop=args.p_drop,
    ).to(device)
    with torch.no_grad():
        b_warmup = next(iter(
            DataLoader(tr_ds, batch_size=min(args.n_batch, len(tr_ds)))
        )).to(device)
        model(b_warmup)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params:,} params  d_edge={d_edge}  "
          f"({'E-BasePlanE' if d_edge > 0 else 'BasePlanE'})")

    # --- optim
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=25, min_lr=1e-6,
    )
    criterion = nn.L1Loss()

    # --- ckpt + CSV log
    os.makedirs(args.save_dir, exist_ok=True)
    slug = args.benchmark.replace("/", "_")
    p_ckpt = os.path.join(args.save_dir, f"polaris_{slug}_best.pt")
    p_csv  = os.path.join(args.save_dir, f"polaris_{slug}_log.csv")
    f_log = open(p_csv, "w", buffering=1)
    writer = csv.writer(f_log)
    writer.writerow(["epoch", "lr", "train_mae", "test_mae", "best_test_mae"])

    # Hold out a small chunk of train for our local "val" MAE (Polaris hides
    # test labels). The official score comes from `bench.evaluate(preds)`.
    n_val = max(1, len(tr_ds) // 5)
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(tr_ds), generator=g).tolist()
    ds_val = [tr_ds[i] for i in perm[:n_val]]
    ds_fit = [tr_ds[i] for i in perm[n_val:]]
    loader_fit = DataLoader(ds_fit, batch_size=args.n_batch, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=args.n_batch)
    print(f"  fit {len(ds_fit)}  val {len(ds_val)}  test {len(te_ds)} (labels hidden)")

    print("\nTraining...")
    print("=" * 70)
    mae_best_val = float("inf")
    i_best = 0
    test_preds_at_best = None
    for i_ep in range(1, args.n_epochs + 1):
        _, mae_fit, _, _ = run_epoch(model, loader_fit, criterion, device,
                                     optimizer=optimizer)
        _, mae_val, _, _ = run_epoch(model, loader_val, criterion, device)
        # Test "MAE" against placeholder zeros is meaningless — we only run
        # the forward pass to capture predictions for Polaris submission.
        _, _, te_preds, _ = run_epoch(model, loader_te, criterion, device)
        scheduler.step(mae_val)

        if mae_val < mae_best_val:
            mae_best_val = mae_val
            i_best = i_ep
            test_preds_at_best = te_preds.numpy()
            torch.save({
                "epoch": i_ep, "model_state": model.state_dict(),
                "best_val_mae": mae_best_val, "args": vars(args),
            }, p_ckpt)

        lr_now = optimizer.param_groups[0]["lr"]
        writer.writerow([i_ep, lr_now, mae_fit, mae_val, mae_best_val])
        if i_ep == 1 or i_ep % 10 == 0:
            print(f"ep {i_ep:03d}  fit {mae_fit:.4f}  val {mae_val:.4f}  "
                  f"best_val {mae_best_val:.4f} @ {i_best}  lr {lr_now:.1e}")

    f_log.close()
    print("=" * 70)
    print(f"best val MAE {mae_best_val:.4f} @ epoch {i_best}  ({p_ckpt})")
    print(f"per-epoch log: {p_csv}")

    # --- Polaris server-side scoring on the best-val test predictions.
    if test_preds_at_best is not None:
        results = bench.evaluate(test_preds_at_best.tolist())
        print("\nPolaris evaluate() on test set:")
        print(results)


if __name__ == "__main__":
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    main()
