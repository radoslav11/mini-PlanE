"""
Simplified PlanE layers without complex configuration flags.
"""

import torch
from   torch                    import nn
from   torch_geometric          import nn as tgnn
import torch_scatter


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for planar orderings."""

    def __init__(self, dim, base_freq=1 / 64):
        super().__init__()
        self.dim = dim
        self.base_freq = base_freq

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [N] containing integer positions
        Returns:
            Tensor of shape [N, dim] with sinusoidal encodings
        """
        device = x.device
        x = x.float().unsqueeze(-1)  # [N, 1]

        # Create frequency bands
        div_term = torch.exp(
            torch.arange(0, self.dim, 2, device=device)
            * -(torch.log(torch.tensor(10000.0)) / self.dim)
        )

        # Compute sinusoidal encodings
        pe = torch.zeros(x.size(0), self.dim, device=device)
        pe[:, 0::2] = torch.sin(x * div_term * self.base_freq)
        pe[:, 1::2] = torch.cos(x * div_term * self.base_freq)

        return pe


class MLP(nn.Module):
    """Simple MLP with LayerNorm and ReLU."""

    def __init__(self, in_dim, out_dim, hidden_factor=2, dropout=0.0):
        super().__init__()
        if hidden_factor <= 0:
            # Simple linear + norm + activation
            self.net = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            # Two-layer MLP
            hidden_dim = in_dim * hidden_factor
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

    def forward(self, x):
        return self.net(x)


class TriconnectedEncoder(nn.Module):
    """
    Encoder for triconnected components (3-connected subgraphs).

    This captures the fundamental planar structure through SPQR tree decomposition.
    """

    def __init__(self, hidden_dim, pe_dim=16, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Learnable embeddings for virtual edges
        self.h_virtual = nn.Parameter(torch.randn(hidden_dim))

        # Positional encoding for canonical orderings
        self.pe = PositionalEncoding(pe_dim)

        # MLP to combine node, edge, and positional features
        self.combine_mlp = MLP(
            2 * hidden_dim + 2 * pe_dim,  # node + edge + 2 PEs
            hidden_dim,
            hidden_factor=2,
            dropout=dropout,
        )

        # Separate MLPs for different component types (S, P, R)
        self.component_mlps = nn.ModuleList(
            [
                MLP(hidden_dim, hidden_dim, hidden_factor=0, dropout=dropout)
                for _ in range(3)
            ]
        )

        self.bn = nn.LayerNorm(hidden_dim)  # Use LayerNorm instead of BatchNorm to avoid batch size issues

    def forward(self, data, h_nodes, h_edges=None):
        """
        Encode triconnected components.

        Args:
            data: PyG Data with SPQR attributes
            h_nodes: Node embeddings [num_nodes, hidden_dim]
            h_edges: Edge embeddings [num_edges, hidden_dim] or None

        Returns:
            Tensor [num_triconnected, hidden_dim]: Component embeddings
        """
        # Read SPQR tree structure
        # spqr_read_from_e contains:
        #   [0]: component_id, [1]: node_u, [2]: node_v,
        #   [3]: edge_id (-1 for virtual edges),
        #   [4]: code1 (position in canonical walk),
        #   [5]: code2 (kappa[code1])
        id_spqr, id_u, id_v, id_e, code1, code2 = data.spqr_read_from_e

        # Initialize edge features
        num_edges_in_components = id_e.size(0)
        h_component_edges = torch.zeros(
            num_edges_in_components, self.hidden_dim, device=h_nodes.device
        )

        # Virtual edges get learnable embedding
        virtual_mask = id_e < 0
        h_component_edges[virtual_mask] = self.h_virtual

        # Real edges get their embeddings
        if h_edges is not None:
            h_component_edges[~virtual_mask] = h_edges[id_e[~virtual_mask]]

        # Encode positional information
        pe1 = self.pe(code1)
        pe2 = self.pe(code2)

        # Combine: node_u + edge + pos1 + pos2
        combined = self.combine_mlp(
            torch.cat([h_nodes[id_u], h_component_edges, pe1, pe2], dim=1)
        )

        # Aggregate to component level
        num_components = data.spqr_batch.size(0)
        component_features = torch_scatter.scatter(
            combined, id_spqr, dim=0, dim_size=num_components, reduce="add"
        )

        # Apply component-type-specific MLPs
        out = torch.zeros(
            num_components, self.hidden_dim, device=h_nodes.device
        )
        for component_type in range(3):  # S, P, R types
            mask = data.spqr_type == component_type
            if mask.any():
                out[mask] = self.component_mlps[component_type](
                    component_features[mask]
                )

        # Batch normalization
        out = self.bn(out)

        return out


class BiconnectedEncoder(nn.Module):
    """
    Encoder for biconnected components (2-connected subgraphs).

    Uses recursive message passing on the SPQR tree.
    """

    def __init__(self, hidden_dim, pe_dim=16, dropout=0.0):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Positional encoding for tree orderings
        self.pe = PositionalEncoding(pe_dim)

        # Message passing on SPQR tree
        self.update_mlp = MLP(
            hidden_dim + pe_dim, hidden_dim, hidden_factor=2, dropout=dropout
        )

        self.post_update_mlp = MLP(
            hidden_dim, hidden_dim, hidden_factor=0, dropout=dropout
        )

        # Read from tree root
        self.read_mlp = MLP(
            hidden_dim, hidden_dim, hidden_factor=2, dropout=dropout
        )

    def forward(self, data, h_triconnected):
        """
        Encode biconnected components from triconnected components.

        Args:
            data: PyG Data with SPQR attributes
            h_triconnected: Triconnected component embeddings

        Returns:
            Tensor [num_biconnected, hidden_dim]: Biconnected embeddings
        """
        h_spqr = h_triconnected.clone()

        # Positional encodings for tree edges
        h_spqr_edge = self.pe(data.spqr_edge_attr)

        # Bottom-up message passing on SPQR tree
        max_order = data.spqr_order.max().item()
        for cur_order in range(max_order + 1):
            # Process nodes at this level
            node_mask = data.spqr_order == cur_order

            # Find edges to children
            edge_mask = node_mask[data.spqr_edge_index[1]]

            if edge_mask.any():
                # Aggregate from children
                src, dst = data.spqr_edge_index[:, edge_mask]

                # Message passing with edge features
                messages = torch.cat(
                    [h_spqr[src], h_spqr_edge[edge_mask]], dim=1
                )

                messages = self.update_mlp(messages)

                # Aggregate messages
                aggregated = torch_scatter.scatter(
                    messages, dst, dim=0, dim_size=h_spqr.size(0), reduce="add"
                )

                # Update nodes
                h_spqr[node_mask] = h_spqr[node_mask] + self.post_update_mlp(
                    aggregated[node_mask]
                )

        # Read from canonical centers to get biconnected representations
        num_biconnected = data.b_batch.size(0)

        # Initialize biconnected features
        h_b = torch.zeros(
            num_biconnected, self.hidden_dim, device=h_spqr.device
        )

        # Read from SPQR tree roots
        src, dst = data.b_read_from_spqr_root
        messages = self.read_mlp(h_spqr[src])
        h_b = torch_scatter.scatter(
            messages, dst, dim=0, dim_size=num_biconnected, reduce="add"
        )

        return h_b


class PlaneLayer(nn.Module):
    """
    Single PlanE layer that aggregates from multiple structural levels.

    This is the core building block of the PlanE architecture.
    """

    def __init__(
        self,
        hidden_dim,
        dropout=0.0,
        use_neighbors=True,
        use_triconnected=True,
        use_biconnected=True,
        use_global_readout=True,
        positional_encoding_dim=16,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_neighbors = use_neighbors
        self.use_triconnected = use_triconnected
        self.use_biconnected = use_biconnected
        self.use_global_readout = use_global_readout

        # Count number of aggregations for combining later
        num_aggs = sum(
            [
                use_neighbors,
                use_triconnected,
                use_biconnected,
                use_global_readout,
            ]
        )

        # 1. Neighbor aggregation (standard GNN)
        if use_neighbors:
            self.neighbor_conv = tgnn.GINConv(
                MLP(hidden_dim, hidden_dim, hidden_factor=2, dropout=dropout),
                train_eps=True,
            )

        # 2. Triconnected component encoder
        if use_triconnected:
            self.tri_encoder = TriconnectedEncoder(
                hidden_dim, positional_encoding_dim, dropout
            )
            self.tri_aggregator = tgnn.GINConv(
                MLP(hidden_dim, hidden_dim, hidden_factor=2, dropout=dropout),
                train_eps=True,
            )

        # 3. Biconnected component encoder
        if use_biconnected:
            self.bi_encoder = BiconnectedEncoder(
                hidden_dim, positional_encoding_dim, dropout
            )
            self.bi_aggregator = tgnn.GINConv(
                MLP(hidden_dim, hidden_dim, hidden_factor=2, dropout=dropout),
                train_eps=True,
            )

        # 4. Global readout
        if use_global_readout:
            self.global_pool = tgnn.global_add_pool
            self.global_mlp = MLP(
                hidden_dim, hidden_dim, hidden_factor=2, dropout=dropout
            )

        # Combine all aggregations
        self.combine_mlp = nn.Sequential(
            nn.Linear(hidden_dim * num_aggs, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, data, h, edge_attr=None):
        """
        Forward pass through PlaneLayer.

        Args:
            data: PyG Data with SPQR preprocessing
            h: Node features [num_nodes, hidden_dim]
            edge_attr: Edge features [num_edges, hidden_dim] or None

        Returns:
            Tensor [num_nodes, hidden_dim]: Updated node features
        """
        aggregations = []

        # 1. Aggregate from neighbors
        if self.use_neighbors:
            h_neighbors = self.neighbor_conv(h, data.edge_index)
            aggregations.append(h_neighbors)

        # 2. Aggregate from triconnected components
        if self.use_triconnected:
            h_tri = self.tri_encoder(data, h, edge_attr)
            h_from_tri = self.tri_aggregator((h_tri, h), edge_index=data.g_read_from_spqr)
            aggregations.append(h_from_tri)

        # 3. Aggregate from biconnected components
        if self.use_biconnected:
            # First compute triconnected (needed for biconnected)
            if not self.use_triconnected:
                h_tri = self.tri_encoder(data, h, edge_attr)

            h_bi = self.bi_encoder(data, h_tri)
            h_from_bi = self.bi_aggregator((h_bi, h), edge_index=data.g_read_from_b)
            aggregations.append(h_from_bi)

        # 4. Aggregate from global readout
        if self.use_global_readout:
            h_global = self.global_pool(h, data.batch)  # [num_graphs, hidden_dim]
            h_global = self.global_mlp(h_global)  # [num_graphs, hidden_dim]
            # Broadcast back to nodes
            h_from_global = h_global[data.batch]  # [num_nodes, hidden_dim]
            aggregations.append(h_from_global)

        # Combine all aggregations
        h_combined = torch.cat(aggregations, dim=1)
        h_new = self.combine_mlp(h_combined)

        return h_new
