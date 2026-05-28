"""Unit tests for the PlanE model on two small planar graphs.

Covers: forward shape, backward gradient flow, and isomorphism invariance
(two node-relabelings of K4 must produce identical model outputs).
"""

import pytest
import torch
from torch_geometric.data import Batch, Data

from plane import PlanE, planar_preprocess


def _k4():
    """K4 (4 nodes, 6 undirected edges) — planar, triconnected, no cut nodes."""
    ei = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 0, 0, 2, 1, 3],
        [1, 0, 2, 1, 3, 2, 0, 3, 2, 0, 3, 1],
    ])
    return Data(x=torch.ones((4, 1)), edge_index=ei, y=torch.tensor([0]))


def _cut_graph():
    """Two triangles sharing node 2 → node 2 is a cut node; two biconn comps."""
    ei = torch.tensor([
        [0, 1, 1, 2, 2, 0, 2, 3, 3, 4, 4, 2],
        [1, 0, 2, 1, 0, 2, 3, 2, 4, 3, 2, 4],
    ])
    return Data(x=torch.ones((5, 1)), edge_index=ei, y=torch.tensor([1]))


@pytest.fixture(scope="module")
def batch_two():
    """Batch of two preprocessed planar graphs (one triconnected + one with cuts)."""
    d_a = planar_preprocess(_k4())
    d_b = planar_preprocess(_cut_graph())
    return Batch.from_data_list([d_a, d_b])


def test_forward_shape(batch_two):
    torch.manual_seed(0)
    model = PlanE(d_node=1, n_cls=3, d_hid=32, n_layers=2).eval()
    out = model(batch_two)
    assert out.shape == (2, 3)
    assert torch.isfinite(out).all()


def test_backward(batch_two):
    torch.manual_seed(0)
    model = PlanE(d_node=1, n_cls=3, d_hid=32, n_layers=2)
    out = model(batch_two)
    loss = torch.nn.functional.cross_entropy(out, batch_two.y)
    loss.backward()
    n_with_grad = sum(
        1 for p in model.parameters()
        if p.grad is not None and p.grad.abs().sum().item() > 0
    )
    # Plenty of trainable params; if any subset doesn't get a gradient we have
    # a disconnected branch in the forward graph.
    assert n_with_grad > 20


def test_isomorphism_invariance():
    """Two permutations of K4 must give identical outputs (perm-invariance)."""
    # Relabel nodes: 0->1, 1->2, 2->3, 3->0.
    perm = torch.tensor([1, 2, 3, 0])
    g_a = _k4()
    g_b = Data(
        x=torch.ones((4, 1)),
        edge_index=perm[g_a.edge_index],
        y=torch.tensor([0]),
    )
    b_a = Batch.from_data_list([planar_preprocess(g_a)])
    b_b = Batch.from_data_list([planar_preprocess(g_b)])

    torch.manual_seed(0)
    model = PlanE(d_node=1, n_cls=4, d_hid=32, n_layers=2).eval()
    out_a = model(b_a)
    out_b = model(b_b)
    # Eval mode + fresh BN running stats (mean 0, var 1) keeps the forward
    # function strictly permutation-invariant up to floating-point noise.
    assert torch.allclose(out_a, out_b, atol=1e-5), (
        f"isomorphism-invariance violated: max diff = "
        f"{(out_a - out_b).abs().max().item():.2e}"
    )
