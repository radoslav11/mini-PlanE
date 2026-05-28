#!/usr/bin/env python
# ruff: noqa: E402 — imports below the sys.path.insert / Sage-env setup
"""Train mini-PlanE on the P3R dataset (paper Section 7.1.2).

P3R: 9 classes of planar 3-regular graphs of size 10, each class containing
50 isomorphic permutations. BasePlanE is expected to reach 100% accuracy.
"""

import argparse
import os
import pickle
import random
import sys
from pathlib import Path

# Paths are anchored at the repo root so the script works from any cwd.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch import nn, optim
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Sage caches under DOT_SAGE / SAGE_CACHE_DIR; point them at a writable local
# directory before importing `plane` (which loads Sage).
d_sage = str(ROOT / ".sage")
os.makedirs(d_sage, exist_ok=True)
os.environ.setdefault("DOT_SAGE", d_sage)
os.environ.setdefault("SAGE_CACHE_DIR", d_sage)

from plane import PlanE, planar_preprocess  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset


class P3RDataset(InMemoryDataset):
    """PyG InMemoryDataset wrapping the upstream P3R pickle + SPQR preprocess."""

    def __init__(self, root, src_path, transform=None, pre_transform=None):
        self.src_path = src_path
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False
        )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    def process(self):
        data_raw = pickle.load(open(self.src_path, "rb"))
        data_out = []
        print(f"Processing {len(data_raw)} graphs (SPQR preprocess)...")
        for d in tqdm(data_raw):
            if d.x is not None:
                d.x = d.x.float()
            data_out.append(self.pre_transform(d))
        d_collated, slices = self.collate(data_out)
        torch.save((d_collated, slices), self.processed_paths[0])


# ---------------------------------------------------------------------------
# Train / eval loops


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`. Trains if `optimizer` is given, else evaluates."""
    is_train = optimizer is not None
    model.train(is_train)
    n_loss = n_correct = n_total = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            n_loss += loss.item() * batch.num_graphs
            n_correct += (out.argmax(1) == batch.y).sum().item()
            n_total += batch.num_graphs
    return n_loss / n_total, n_correct / n_total


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
    # Defaults match the upstream P3R sweep config (plane.yaml):
    #   100 epochs, batch 128, lr 1e-3, 2 layers, hidden 64, PE 16, dropout 0.
    p = argparse.ArgumentParser(description="Train PlanE on the P3R dataset")
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--n-batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--d-hid", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-pe", type=int, default=16)
    p.add_argument("--p-drop", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--n-patience",
        type=int,
        default=0,
        help="Early stop patience (0 = disabled)",
    )
    p.add_argument("--scheduler", choices=["none", "cos"], default="cos")
    p.add_argument("--save-dir", type=str, default=str(ROOT / ".checkpoints"))
    p.add_argument("--cache-dir", type=str, default=str(ROOT / ".dataset/P3R"))
    p.add_argument(
        "--src-pickle",
        type=str,
        default=str(ROOT.parent / "PlanE" / ".dataset_src" / "P3R.pkl"),
        help="Path to upstream P3R.pkl (from the ZZYSonny/PlanE repo).",
    )
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--eval-only", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    p_ckpt = args.checkpoint or os.path.join(args.save_dir, "p3r_best.pt")

    # ---- data
    print("Loading P3R dataset...")
    print("=" * 70)
    dataset = P3RDataset(
        root=args.cache_dir,
        src_path=args.src_pickle,
        pre_transform=planar_preprocess,
    )
    print(
        f"  n_graphs: {len(dataset)}  n_cls: {dataset.num_classes}  "
        f"d_node: {dataset.num_node_features}"
    )

    # Fold 0 of a 10-fold split (labels are pre-shuffled in the pickle).
    n_graphs = len(dataset)
    n_fold = n_graphs // 10
    ds_test = dataset[list(range(0, n_fold))]
    ds_train = dataset[list(range(n_fold, n_graphs))]
    print(f"  train: {len(ds_train)}  test: {len(ds_test)}")

    loader_train = DataLoader(ds_train, batch_size=args.n_batch, shuffle=True)
    loader_test = DataLoader(ds_test, batch_size=args.n_batch)

    # ---- model
    model = PlanE(
        d_node=dataset.num_node_features,
        n_cls=dataset.num_classes,
        d_hid=args.d_hid,
        n_layers=args.n_layers,
        d_pe=args.d_pe,
        p_drop=args.p_drop,
    ).to(device)

    # Materialize lazy submodules (PlaneLayer.mlp_combine etc.) before counting
    # parameters. Use an unshuffled loader so RNG isn't consumed here.
    with torch.no_grad():
        b_warmup = next(
            iter(
                DataLoader(
                    ds_train, batch_size=min(args.n_batch, len(ds_train))
                )
            )
        ).to(device)
        model(b_warmup)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params:,} params")

    # ---- optim + (optional) eval-only
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
        if args.scheduler == "cos"
        else None
    )

    if args.eval_only:
        if not os.path.exists(p_ckpt):
            print(f"Checkpoint not found: {p_ckpt}")
            return
        print(f"Evaluating: {p_ckpt}")
        model.load_state_dict(
            torch.load(p_ckpt, map_location=device)["model_state"]
        )
        loss_te, acc_te = run_epoch(model, loader_test, criterion, device)
        print(f"  test loss {loss_te:.4f}  test acc {acc_te:.4f}")
        return

    # ---- train
    print("\nTraining...")
    print("=" * 70)
    acc_best = 0.0
    i_best = 0
    n_stale = 0
    for i_ep in range(1, args.n_epochs + 1):
        loss_tr, acc_tr = run_epoch(
            model, loader_train, criterion, device, optimizer=optimizer
        )
        loss_te, acc_te = run_epoch(model, loader_test, criterion, device)
        if scheduler is not None:
            scheduler.step()

        if acc_te > acc_best:
            acc_best, i_best, n_stale = acc_te, i_ep, 0
            torch.save(
                {
                    "epoch": i_ep,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_test_acc": acc_best,
                    "args": vars(args),
                },
                p_ckpt,
            )
        else:
            n_stale += 1

        if i_ep == 1 or i_ep % 10 == 0:
            print(
                f"ep {i_ep:03d}  tr {loss_tr:.4f}/{acc_tr:.4f}  "
                f"te {loss_te:.4f}/{acc_te:.4f}  "
                f"best {acc_best:.4f} @ {i_best}"
            )

        if args.n_patience > 0 and n_stale >= args.n_patience:
            print(f"Early stop @ ep {i_ep} (stale={n_stale})")
            break

    print("=" * 70)
    print(f"best test acc {acc_best:.4f} @ epoch {i_best}  ({p_ckpt})")


if __name__ == "__main__":
    main()
