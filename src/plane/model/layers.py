"""PlanE layers (Section 5 of Dimitrov et al., 2023, NeurIPS).

Implements BasePlanE: TriEnc, BiEnc, CutEnc, and the per-layer combine that
aggregates 1-hop neighbors, triconnected components, biconnected components, a
global readout, and the cut subtree representation.

Tensor shape suffixes (Hungarian-style; the suffix after `__` lists dims in
order):
  N   nodes in the input graph(s), batched
  E   edges in the input graph(s), batched
  G   number of input graphs in the batch
  S   SPQR (triconnected) components, batched
  B   biconnected components, batched
  T   edges of the SPQR forest, batched
  K   entries in `spqr_read_from_e` (one per (component, cycle-edge))
  D   hidden dim (`d_hid`)
  P   positional-encoding dim (`d_pe`)
  C   number of output classes (`n_cls`)
So e.g. `h_g__N_D` is the node-feature tensor of shape [N, D], `h_b__B_D` is one
vector per biconnected component, and `id_u__K` is a long tensor of node ids.

Scalar hyperparameter names: `d_hid, d_pe, p_drop, n_layers` etc.
"""

import torch
import torch_scatter
from torch import nn
from torch_geometric import nn as tgnn


def PosEnc(d):
    """Sinusoidal positional encoding p_x (paper Section 5, ref [57]).

    Uses the PyG implementation with `base_freq = 1/64` per Appendix D.4
    ("periodicity of 64"). Stateless apart from a precomputed frequency buffer.
    """
    return tgnn.PositionalEncoding(d, base_freq=1 / 64)


def make_mlp(d_in, d_out, factor_hid=2, p_drop=0.0, norm="batch_norm"):
    """Two-layer MLP with ReLU. `norm` is "batch_norm" or "none"."""

    def Norm(d_feat):
        return nn.Identity() if norm == "none" else nn.BatchNorm1d(d_feat)

    d_mid = d_in * factor_hid
    return nn.Sequential(
        nn.Linear(d_in, d_mid),
        Norm(d_mid),
        nn.ReLU(),
        nn.Dropout(p_drop),
        nn.Linear(d_mid, d_out),
        Norm(d_out),
        nn.ReLU(),
        nn.Dropout(p_drop),
    )


class TriEnc(nn.Module):
    """Triconnected-component encoder. Paper Section 5, TRIENC.

    For each SPQR component C, the canonical Weinberg walk visits nodes
    omega_0..omega_{k-1} (each edge traversed twice). For position i let
    kappa_i be the first-visit-order of omega_i. Then

        bh_C = MLP_type(C) ( sum over i of
                     MLP( h[omega_i] || h_edge_i || PE(kappa_i) || PE(i) ) )

    `h_edge_i` is the input edge feature when the cycle edge is real, or one
    of two learnable placeholders (`h_virtual` for SPQR virtual edges, `h_edge`
    for real edges with no provided edge feature). A separate output MLP is
    applied per SPQR type (S/Q -> 0, P -> 1, R -> 2), followed by BatchNorm.
    """

    def __init__(self, d_hid, d_pe=16, p_drop=0.0):
        super().__init__()
        self.d_hid = d_hid
        self.h_virtual__D = nn.Parameter(torch.randn(d_hid))
        self.h_edge__D = nn.Parameter(torch.randn(d_hid))
        self.pe = PosEnc(d_pe)
        self.mlp_pre = make_mlp(
            2 * d_hid + 2 * d_pe,
            d_hid,
            factor_hid=2,
            p_drop=p_drop,
            norm="batch_norm",
        )
        self.mlp_per_type = nn.ModuleList(
            [
                make_mlp(
                    d_hid, d_hid, factor_hid=2, p_drop=p_drop, norm="none"
                )
                for _ in range(3)
            ]
        )
        self.bn_out = nn.BatchNorm1d(d_hid)

    def forward(self, data, h_g__N_D, h_e__E_D=None):
        # spqr_read_from_e rows = [id_spqr, id_u, id_v, id_e, code1, code2]
        # All have length K (one entry per (component, cycle-edge)).
        id_spqr__K, id_u__K, _, id_e__K, code1__K, code2__K = (
            data.spqr_read_from_e
        )

        # Per-cycle-edge feature tensor.
        h_cycle__K_D = torch.zeros(
            id_e__K.size(0), self.d_hid, device=h_g__N_D.device
        )
        mask_virtual__K = id_e__K < 0
        h_cycle__K_D[mask_virtual__K] = self.h_virtual__D
        if h_e__E_D is None:
            h_cycle__K_D[~mask_virtual__K] = self.h_edge__D
        else:
            h_cycle__K_D[~mask_virtual__K] = h_e__E_D[
                id_e__K[~mask_virtual__K]
            ]

        # Inner MLP on (h[omega_i] || h_edge_i || PE(kappa_i) || PE(i)).
        h_pre__K_D = self.mlp_pre(
            torch.cat(
                [
                    h_g__N_D[id_u__K],
                    h_cycle__K_D,
                    self.pe(code2__K),
                    self.pe(code1__K),
                ],
                dim=1,
            )
        )

        # Sum over walk positions within each component; per-type outer MLP; BN.
        n_S = data.spqr_batch.size(0)
        h_sum__S_D = torch_scatter.scatter(
            h_pre__K_D, id_spqr__K, dim=0, dim_size=n_S, reduce="add"
        )
        h_out__S_D = torch.zeros(n_S, self.d_hid, device=h_g__N_D.device)
        for i_type in range(3):
            mask_type__S = data.spqr_type == i_type
            if mask_type__S.any():
                h_out__S_D[mask_type__S] = self.mlp_per_type[i_type](
                    h_sum__S_D[mask_type__S]
                )
        return self.bn_out(h_out__S_D)


class BiEnc(nn.Module):
    """Biconnected-component encoder. Paper Section 5, BIENC.

    Bottom-up recurrence on the SPQR tree gamma = SPQR(B):

        eh_subtree(C) = MLP( bh_C + sum over C' in children(C) of
                             MLP( eh_subtree(C') || PE(theta(C, C')) ) )

    and read the root: eh_B = eh_subtree(root). `update` uses GINEConv with
    Identity NN (so each message is implicitly ReLU(src + PE(theta)) after the
    next ReLU). `read` aggregates SPQR-tree-root reps into per-biconnected
    tensors via GINConv on the bipartite edges `b_read_from_spqr_root`.
    """

    def __init__(self, d_hid, p_drop=0.0):
        super().__init__()
        self.d_hid = d_hid
        # PE has width D so GINEConv can sum it with the source feature.
        self.pe = PosEnc(d_hid)
        self.update = tgnn.GINEConv(nn.Identity())
        self.mlp_post = make_mlp(
            d_hid,
            d_hid,
            factor_hid=2,
            p_drop=p_drop,
            norm="none",
        )
        self.read = tgnn.GINConv(
            make_mlp(
                d_hid, d_hid, factor_hid=2, p_drop=p_drop, norm="batch_norm"
            )
        )

    def forward(self, data, h_spqr__S_D):
        # Work buffer holds in-flight subtree reps while we sweep the tree.
        h_work__S_D = h_spqr__S_D.clone()
        pe_theta__T_D = self.pe(data.spqr_edge_attr)

        # Process SPQR tree nodes bottom-up by precomputed `spqr_order`.
        n_orders = data.spqr_order.max().item() + 1
        for i_order in range(n_orders):
            mask_node__S = data.spqr_order == i_order
            mask_edge__T = mask_node__S[data.spqr_edge_index[1]]
            h_new__S_D = self.update(
                h_work__S_D.clone(),
                edge_index=data.spqr_edge_index[:, mask_edge__T],
                edge_attr=pe_theta__T_D[mask_edge__T],
            )
            h_work__S_D[mask_node__S] = self.mlp_post(h_new__S_D[mask_node__S])

        # Read one vector per biconnected component, from its SPQR root.
        n_B = data.b_batch.size(0)
        h_b_init__B_D = torch.zeros(n_B, self.d_hid, device=h_work__S_D.device)
        return self.read(
            (h_work__S_D, h_b_init__B_D), data.b_read_from_spqr_root
        )


class CutEnc(nn.Module):
    """Cut-subtree encoder. Paper Section 5, CUTENC.

    For each cut node u of the Block-Cut tree, compute h_delta(u) for the
    subtree rooted at u:

        h_delta(u) = MLP( h_u + sum over B in children(u) of
                          MLP( eh_B + sum over v in children(B) of h_delta(v) ) )

    Realized by alternating updates between B (biconnected) nodes and C (cut)
    nodes in increasing `b_order` / `c_order`. The returned per-node tensor is
    zero at non-cut nodes.
    """

    def __init__(self, d_hid, p_drop=0.0):
        super().__init__()
        self.d_hid = d_hid
        self.update = tgnn.GINConv(nn.Identity())
        self.mlp_post = make_mlp(
            d_hid,
            d_hid,
            factor_hid=2,
            p_drop=p_drop,
            norm="none",
        )

    def forward(self, data, h_g__N_D, h_b__B_D):
        n_orders = (
            max(data.b_order.max().item(), data.c_order.max().item()) + 1
        )

        # Work buffers; messages flow back-and-forth between B and C nodes.
        h_g_work__N_D = h_g__N_D.clone()
        h_b_work__B_D = h_b__B_D.clone()

        for i_order in range(n_orders):
            # B nodes at this level absorb messages from their child cut nodes.
            mask_b__B = data.b_order == i_order
            mask_cb__T = mask_b__B[data.cb_edge_index[1]]
            h_b_work__B_D = self.update(
                (h_g_work__N_D, h_b_work__B_D),
                edge_index=data.cb_edge_index[:, mask_cb__T],
            )
            h_b_work__B_D[mask_b__B] = self.mlp_post(h_b_work__B_D[mask_b__B])

            # C nodes at this level absorb messages from their child B nodes.
            mask_c__N = data.c_order == i_order
            mask_bc__T = mask_c__N[data.bc_edge_index[1]]
            h_g_work__N_D = self.update(
                (h_b_work__B_D, h_g_work__N_D),
                edge_index=data.bc_edge_index[:, mask_bc__T],
            )
            h_g_work__N_D[mask_c__N] = self.mlp_post(
                h_g_work__N_D[mask_c__N]
            )

        # Output is zero everywhere except at cut nodes (paper: h_delta(u) is
        # only defined when u is a cut node; otherwise it contributes nothing).
        mask_cut__N = data.c_order >= 0
        h_delta__N_D = torch.zeros_like(h_g__N_D)
        h_delta__N_D[mask_cut__N] = h_g_work__N_D[mask_cut__N]
        return h_delta__N_D


class PlaneLayer(nn.Module):
    """One BasePlanE layer (Section 5, UPDATE).

    Concatenates five per-node aggregations and projects back to D:

        h_u_next = f(
            g1( h_u + sum over neighbors of h_v )         # 1-hop neighbors
         || g2( sum over all v of h_v )                    # global readout
         || g3( h_u + sum over C containing u of bh_C )    # triconnected
         || g4( h_u + sum over B containing u of eh_B )    # biconnected
         || h_delta(u)                                     # cut subtree
        )

    g1..g4 are GINConv (`train_eps=True`); the global readout uses
    DeepSetsAggregation. f is Linear -> BatchNorm -> ReLU -> Dropout.
    """

    def __init__(self, d_hid, p_drop=0.0, d_pe=16, d_edge=0):
        super().__init__()
        self.d_hid = d_hid
        self.has_edge_feat = d_edge > 0

        # All per-aggregation MLPs share shape and use no internal norm
        # (the layer's `mlp_combine` provides BatchNorm after concat).
        def gin_mlp():
            return make_mlp(
                d_hid, d_hid, factor_hid=2, p_drop=p_drop, norm="none"
            )

        # Neighbor aggregation: GINEConv when edge features are present so
        # messages depend on (h_src + h_edge); plain GINConv otherwise.
        if self.has_edge_feat:
            self.aggr_neigh = tgnn.GINEConv(gin_mlp(), train_eps=True)
        else:
            self.aggr_neigh = tgnn.GINConv(gin_mlp(), train_eps=True)
        self.aggr_spqr = tgnn.GINConv(gin_mlp(), train_eps=True)
        self.aggr_b = tgnn.GINConv(gin_mlp(), train_eps=True)

        self.enc_spqr = TriEnc(d_hid, d_pe, p_drop)
        self.enc_b = BiEnc(d_hid, p_drop)
        self.enc_gr = tgnn.DeepSetsAggregation(nn.Identity(), gin_mlp())
        self.enc_cut = CutEnc(d_hid, p_drop)

        # f on the concatenation of 5 per-node aggregations.
        self.mlp_combine = nn.Sequential(
            nn.Linear(d_hid * 5, d_hid),
            nn.BatchNorm1d(d_hid),
            nn.ReLU(),
            nn.Dropout(p_drop),
        )

    def forward(self, data, h_g__N_D, h_e__E_D=None):
        # g1: 1-hop neighbors (GINEConv consumes edge features when present).
        if self.has_edge_feat:
            h_neigh__N_D = self.aggr_neigh(
                h_g__N_D, data.edge_index, edge_attr=h_e__E_D
            )
        else:
            h_neigh__N_D = self.aggr_neigh(h_g__N_D, data.edge_index)

        # g2: global readout, broadcast back to nodes
        h_gr__G_D = self.enc_gr(h_g__N_D, data.batch, dim_size=data.num_graphs)
        h_gr__N_D = h_gr__G_D[data.batch]

        # g3: triconnected components
        h_spqr__S_D = self.enc_spqr(data, h_g__N_D, h_e__E_D)
        h_from_t__N_D = self.aggr_spqr(
            (h_spqr__S_D, h_g__N_D), edge_index=data.g_read_from_spqr
        )

        # g4: biconnected components
        h_b__B_D = self.enc_b(data, h_spqr__S_D)
        h_from_b__N_D = self.aggr_b(
            (h_b__B_D, h_g__N_D), edge_index=data.g_read_from_b
        )

        # Cut subtree representation (zero at non-cut nodes).
        h_cut__N_D = self.enc_cut(data, h_g__N_D, h_b__B_D)

        return self.mlp_combine(
            torch.cat(
                [
                    h_neigh__N_D,
                    h_from_t__N_D,
                    h_from_b__N_D,
                    h_gr__N_D,
                    h_cut__N_D,
                ],
                dim=1,
            )
        )
