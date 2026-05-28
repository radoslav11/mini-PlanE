"""PlanE model (Section 5 of Dimitrov et al., 2023).

Stacks BasePlanE PlaneLayers and reads out a graph-level prediction by
concatenating per-layer sum-pooled node embeddings (jumping knowledge):

    z_G = MLP( || over layers l of sum over u in V_G of h_u_l )

See `plane.model.layers` for the shape-suffix convention (N nodes, E edges,
G graphs, S SPQR components, B biconnected, D hidden, C classes).
"""

import torch
from torch import nn
from torch_geometric import nn as tgnn

from plane.model.layers import PlaneLayer


class PlanE(nn.Module):
    """PlanE: Representation Learning over Planar Graphs.

    Expects PyG `Data` objects preprocessed with `planar_preprocess`.

    Args:
        d_node    : input node feature dim
        n_cls     : output dim (classes or regression targets)
        d_edge    : input edge feature dim (0 = no edge features)
        d_hid     : hidden dim
        n_layers  : number of PlaneLayers (paper P3R: 2)
        p_drop    : dropout probability
        d_pe      : positional-encoding dim inside TriEnc (paper: 16)
    """

    def __init__(
        self,
        d_node,
        n_cls,
        d_edge=0,
        d_hid=64,
        n_layers=2,
        p_drop=0.0,
        d_pe=16,
    ):
        super().__init__()
        self.d_node = d_node
        self.d_edge = d_edge
        self.d_hid = d_hid
        self.n_cls = n_cls
        self.n_layers = n_layers
        self.p_drop = p_drop

        self.embed_node = nn.Linear(max(1, d_node), d_hid)
        self.embed_edge = nn.Linear(d_edge, d_hid) if d_edge > 0 else None

        self.layers = nn.ModuleList(
            [
                PlaneLayer(
                    d_hid=d_hid, p_drop=p_drop, d_pe=d_pe, d_edge=d_edge
                )
                for _ in range(n_layers)
            ]
        )

        # Jumping-knowledge readout: concat per-layer graph reprs -> MLP head.
        self.pool = tgnn.SumAggregation()
        self.head = nn.Sequential(
            nn.Linear(d_hid * n_layers, 2 * d_hid),
            nn.BatchNorm1d(2 * d_hid),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(2 * d_hid, n_cls),
        )

    def forward(self, data):
        # data.x   shape [N, d_node]   (with d_node typically = 1 or small)
        # data.batch shape [N]         (graph id per node)
        # The pool reduces N -> G via data.batch.
        h_g__N_D = self.embed_node(data.x)
        h_e__E_D = (
            self.embed_edge(data.edge_attr)
            if self.embed_edge is not None
            else None
        )

        per_layer__list_N_D = []
        for layer in self.layers:
            h_g__N_D = layer(data, h_g__N_D, h_e__E_D)
            per_layer__list_N_D.append(h_g__N_D)

        # Each pool: [N, D] -> [G, D]; concat over L layers -> [G, L*D].
        h_graph__G_LD = torch.cat(
            [
                self.pool(h__N_D, data.batch, dim_size=data.num_graphs)
                for h__N_D in per_layer__list_N_D
            ],
            dim=1,
        )
        # head: [G, L*D] -> [G, C]
        return self.head(h_graph__G_LD)

    def __repr__(self):
        return (
            f"PlanE(d_node={self.d_node}, d_edge={self.d_edge}, "
            f"d_hid={self.d_hid}, n_cls={self.n_cls}, "
            f"n_layers={self.n_layers}, p_drop={self.p_drop})"
        )
