#!/usr/bin/env python
# ruff: noqa: E402
"""Blend PlanE predictions with a classical descriptor/fingerprint model.

On small ADMET tasks the top *reproducible* TDC entries (MapLight, CaliciBoost)
are gradient boosting on molecular descriptors/fingerprints. A message-passing
GNN like PlanE sees only graph topology + atom/bond types; global
physicochemical descriptors (MolWt, LogP, TPSA, H-bond counts, ...) are
complementary. This script measures whether blending helps.

It reports, on the Polaris test set (via bench.evaluate):
  - descriptor model alone
  - PlanE alone (ensemble of the given checkpoints)
  - blend = a*PlanE + (1-a)*descriptor, for a in a sweep
plus the descriptor model's own 5-fold CV MAE on train (an honest internal
estimate). alpha=0.5 is the pre-registered (non-test-tuned) blend; the full
sweep is shown for insight only. No submission.
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
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold

d_sage = str(ROOT / ".sage")
os.makedirs(d_sage, exist_ok=True)
os.environ.setdefault("DOT_SAGE", d_sage)
os.environ.setdefault("SAGE_CACHE_DIR", d_sage)

from train_polaris import N_ATOM_TYPE  # noqa: F401  (ensures sage import order)
from ensemble_polaris import predict_one
from submit_polaris import slugify_name

RDLogger.DisableLog("rdApp.*")  # pyright: ignore[reportAttributeAccessIssue]

_DESC_FNS = [fn for _, fn in Descriptors._descList]
_N_FP = 1024


def featurize(smi):
    """217 RDKit descriptors + 1024-bit Morgan(r=2) -> 1-D float vector."""
    mol = Chem.MolFromSmiles(smi)
    desc = np.array([fn(mol) for fn in _DESC_FNS], dtype=np.float64)
    desc[~np.isfinite(desc)] = 0.0
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=_N_FP)
    arr = np.zeros(_N_FP, dtype=np.float64)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return np.concatenate([desc, arr])


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def main():
    p = argparse.ArgumentParser(description="Blend PlanE with a descriptor model")
    p.add_argument("--benchmark", type=str, default="tdcommons/caco2-wang")
    p.add_argument("--ckpt", action="append", required=True,
                   help="PlanE checkpoint(s) to ensemble (repeatable).")
    p.add_argument("--cache-dir", type=str, default=str(ROOT / ".dataset" / "polaris"))
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    # Submission (optional). alpha is the pre-registered blend weight.
    p.add_argument("--owner", type=str, default=None,
                   help="If set, upload the alpha-blend result under this owner.")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Blend weight on PlanE for the submitted prediction.")
    p.add_argument("--name", type=str, default="mini-PlanE-Seed-Descriptor")
    p.add_argument("--code-url", type=str,
                   default="https://github.com/radoslav11/mini-PlanE")
    p.add_argument("--paper-url", type=str,
                   default="https://arxiv.org/abs/2307.01180")
    args = p.parse_args()
    device = torch.device(args.device)

    bench = po.load_benchmark(args.benchmark)
    train, test = bench.get_train_test_split()
    train_items = [(smi, float(y)) for smi, y in train]
    test_smis = list(test.inputs)  # pyright: ignore[reportAttributeAccessIssue]
    test_items = [(smi, 0.0) for smi in test_smis]
    y_tr = np.array([y for _, y in train_items])
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std()) + 1e-9
    print(f"benchmark {args.benchmark}  train {len(train_items)}  test {len(test_smis)}")

    # --- descriptor features
    print("featurizing (descriptors + Morgan)...")
    X_tr = np.stack([featurize(smi) for smi, _ in train_items])
    X_te = np.stack([featurize(smi) for smi in test_smis])

    # --- descriptor model: honest 5-fold CV MAE on train (out-of-fold)
    oof = np.zeros(len(y_tr))
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=0)
    for tr_idx, va_idx in kf.split(X_tr):
        m = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=0,
        )
        m.fit(X_tr[tr_idx], y_tr[tr_idx])
        oof[va_idx] = m.predict(X_tr[va_idx])
    desc_cv_mae = mae(oof, y_tr)
    print(f"descriptor model: {args.n_folds}-fold CV MAE on train = {desc_cv_mae:.4f}")

    # refit on full train -> test predictions
    m_full = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        l2_regularization=1.0, random_state=0,
    )
    m_full.fit(X_tr, y_tr)
    desc_test = m_full.predict(X_te)

    # --- PlanE ensemble test predictions
    plane_preds = []
    for c in args.ckpt:
        if not os.path.exists(c):
            print(f"  skip missing {c}")
            continue
        plane_preds.append(predict_one(
            c, args.benchmark, train_items, test_items,
            args.cache_dir, y_mean, y_std, device,
        ))
    plane_test = np.mean(np.stack(plane_preds), axis=0)

    # --- scores on the real test set (via Polaris)
    def test_mae(pred):
        r = bench.evaluate(pred.tolist())
        return float(r.results.to_dict("records")[0]["Score"])

    s_desc  = test_mae(desc_test)
    s_plane = test_mae(plane_test)
    print(f"\ndescriptor-only test MAE : {s_desc:.4f}")
    print(f"PlanE-only test MAE      : {s_plane:.4f}  ({len(plane_preds)} ckpts)")

    print("\nblend a*PlanE + (1-a)*descriptor  (test MAE):")
    best_a, best_s = None, 1e9
    for a in [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]:
        s = test_mae(a * plane_test + (1 - a) * desc_test)
        flag = "  <- a=0.5 (pre-registered)" if abs(a - 0.5) < 1e-9 else ""
        print(f"  a={a:.2f}  {s:.4f}{flag}")
        if s < best_s:
            best_s, best_a = s, a
    print(f"\nbest sweep point: a={best_a:.2f} -> {best_s:.4f} "
          f"(test-peeked, insight only)")
    print(f"banked submitted best (C+D PlanE) = 0.3010; descriptor CV = {desc_cv_mae:.4f}")

    # --- optional submission of the pre-registered alpha blend
    if args.owner:
        blend = args.alpha * plane_test + (1 - args.alpha) * desc_test
        results = bench.evaluate(blend.tolist())
        results.name = slugify_name(args.name)
        results.description = (
            f"{args.alpha:.0%} PlanE + {1 - args.alpha:.0%} descriptor blend. "
            "PlanE side is barely tuned (rich rdkit features + z-scored target, "
            "2-seed ensemble); the descriptor side is HistGradientBoosting on "
            "217 RDKit descriptors + 1024-bit Morgan(r=2) fingerprint. The "
            "descriptor model carries most of the signal."
        )
        results.tags = ["plane", "descriptor-blend", "gnn", "admet", "hybrid"]

        # Reference the repo + paper. Newer (V2) results take a Model artifact
        # with code_url/report_url; older (V1) results expose github_url /
        # paper_url directly. Set whichever the result schema supports.
        fields = type(results).model_fields
        if "github_url" in fields:
            results.github_url = args.code_url
        if "paper_url" in fields:
            results.paper_url = args.paper_url
        if "model" in fields:
            from polaris.model import Model
            results.model = Model(
                name=slugify_name(args.name),
                description="mini-PlanE (planar GNN) blended with an RDKit "
                            "descriptor/fingerprint gradient-boosting model.",
                code_url=args.code_url,
                report_url=args.paper_url,
                tags=["plane", "spqr", "descriptor-blend"],
            )

        score = float(results.results.to_dict("records")[0]["Score"])
        print(f"\nsubmitting alpha={args.alpha} blend (test MAE {score:.4f}) "
              f"as owner={args.owner!r}, name={slugify_name(args.name)!r}...")
        results.upload_to_hub(owner=args.owner)
        print("done.")


if __name__ == "__main__":
    main()
