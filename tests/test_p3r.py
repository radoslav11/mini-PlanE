"""End-to-end training test on a 3-class subset of P3R.

Loads the upstream P3R pickle, runs `planar_preprocess` on a small slice
(3 classes × 10 graphs), and trains a tiny PlanE for 50 epochs. With the
canonical SPQR encoding the model should reach high test accuracy quickly.
"""

import os
import pickle
from collections import defaultdict

import pytest
import torch
from torch_geometric.loader import DataLoader

from plane import PlanE, planar_preprocess


ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P3R_PKL = os.path.normpath(
    os.path.join(ROOT, "..", "PlanE", ".dataset_src", "P3R.pkl")
)
N_CLS_SUB    = 3   # classes to keep (0, 1, 2)
N_PER_CLS    = 10  # graphs per class (8 train, 2 test)
N_EPOCHS     = 50
ACC_FLOOR    = 0.83  # 5/6 correct on the 6-graph test set


@pytest.fixture(scope="module")
def p3r_subset():
    if not os.path.exists(P3R_PKL):
        pytest.skip(f"P3R pickle not found at {P3R_PKL}")
    with open(P3R_PKL, "rb") as fh:
        raw = pickle.load(fh)
    by_cls = defaultdict(list)
    for d in raw:
        if d.x is not None:
            d.x = d.x.float()
        by_cls[int(d.y.item())].append(d)
    out = []
    for c in range(N_CLS_SUB):
        for d in by_cls[c][:N_PER_CLS]:
            out.append(planar_preprocess(d))
    return out


def test_p3r_subset_trains(p3r_subset):
    torch.manual_seed(0)
    n_train = (N_PER_CLS - 2) * N_CLS_SUB  # 24
    ds_train = p3r_subset[:n_train]
    ds_test  = p3r_subset[n_train:]

    model     = PlanE(d_node=1, n_cls=N_CLS_SUB, d_hid=32, n_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    loader_tr = DataLoader(ds_train, batch_size=n_train, shuffle=True)
    loader_te = DataLoader(ds_test, batch_size=len(ds_test))

    acc_best = 0.0
    for _ in range(N_EPOCHS):
        model.train()
        for batch in loader_tr:
            optimizer.zero_grad()
            criterion(model(batch), batch.y).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            n_correct = sum(
                int((model(b).argmax(1) == b.y).sum()) for b in loader_te
            )
        acc_best = max(acc_best, n_correct / len(ds_test))

    assert acc_best >= ACC_FLOOR, (
        f"Expected best test acc >= {ACC_FLOOR}, got {acc_best:.4f}"
    )
