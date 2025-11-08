"""
Simplified PlanE model with sensible defaults.
"""

from   plane.layers             import PlaneLayer
import torch
from   torch                    import nn
from   torch_geometric          import nn as tgnn


class PlanE(nn.Module):
    """
    Simplified PlanE: Representation Learning over Planar Graphs

    A complete and efficient graph neural network for planar graphs based on the
    Hopcroft-Tarjan planar graph isomorphism algorithm.

    Args:
        num_node_features (int): Number of input node features
        num_edge_features (int, optional): Number of input edge features. Default: 0
        hidden_dim (int): Hidden dimension size. Default: 64
        num_classes (int): Number of output classes for classification. Default: 2
        num_layers (int): Number of PlanE layers. Default: 3
        dropout (float): Dropout probability. Default: 0.0
        categorical_node_features (bool): If True, use Embedding for node features (categorical).
                                          If False, use Linear (continuous). Default: False
        categorical_edge_features (bool): If True, use Embedding for edge features (categorical).
                                          If False, use Linear (continuous). Default: False
        use_neighbors (bool): Aggregate from 1-hop neighbors (like GNN). Default: True
        use_triconnected (bool): Aggregate from triconnected components. Default: True
        use_biconnected (bool): Aggregate from biconnected components. Default: True
        use_global_readout (bool): Use global graph readout. Default: True
        positional_encoding_dim (int): Dimension for positional encodings. Default: 16
        task (str): 'classification' or 'regression'. Default: 'classification'

    Example:
        >>> model = PlanE(
        ...     num_node_features=1,
        ...     num_classes=4,
        ...     hidden_dim=64,
        ...     num_layers=3
        ... )
        >>> output = model(data)  # data is PyG Data object with SPQR preprocessing
    """

    def __init__(
        self,
        num_node_features,
        num_edge_features=0,
        hidden_dim=64,
        num_classes=2,
        num_layers=3,
        dropout=0.0,
        categorical_node_features=False,
        categorical_edge_features=False,
        use_neighbors=True,
        use_triconnected=True,
        use_biconnected=True,
        use_global_readout=True,
        positional_encoding_dim=16,
        task="classification",
    ):
        super().__init__()

        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.dropout = dropout
        self.task = task

        # Node feature embedding
        if categorical_node_features:
            self.node_embed = nn.Embedding(num_node_features, hidden_dim)
        else:
            self.node_embed = nn.Linear(num_node_features, hidden_dim)

        # Edge feature embedding (if needed)
        if num_edge_features > 0:
            if categorical_edge_features:
                self.edge_embed = nn.Embedding(num_edge_features, hidden_dim)
            else:
                self.edge_embed = nn.Linear(num_edge_features, hidden_dim)
        else:
            self.edge_embed = None

        # PlanE layers
        self.layers = nn.ModuleList(
            [
                PlaneLayer(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    use_neighbors=use_neighbors,
                    use_triconnected=use_triconnected,
                    use_biconnected=use_biconnected,
                    use_global_readout=use_global_readout,
                    positional_encoding_dim=positional_encoding_dim,
                )
                for _ in range(num_layers)
            ]
        )

        # Graph-level pooling
        self.pool = tgnn.global_add_pool

        # Output MLP
        # We concatenate representations from all layers
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim * num_layers, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

        # Task-specific activation
        if task == "classification":
            self.final_activation = nn.Identity()  # Use with CrossEntropyLoss
        else:
            self.final_activation = nn.Identity()  # For regression

    def forward(self, data):
        """
        Forward pass through the PlanE model.

        Args:
            data: PyTorch Geometric Data object with SPQR preprocessing.
                  Must contain: x, edge_index, edge_attr, batch
                  And SPQR decomposition attributes (added by preprocessing)

        Returns:
            Tensor: Output predictions of shape [num_graphs, num_classes]
        """
        # Embed node features
        h = self.node_embed(data.x)

        # Embed edge features if available
        if self.edge_embed is not None and hasattr(data, "edge_attr"):
            edge_attr = self.edge_embed(data.edge_attr)
        else:
            edge_attr = None

        # Store representations from each layer
        layer_outputs = []

        # Pass through PlanE layers
        for layer in self.layers:
            h = layer(data, h, edge_attr)
            layer_outputs.append(h)

        # Pool node representations to graph level for each layer
        graph_reprs = []
        for h_layer in layer_outputs:
            graph_repr = self.pool(h_layer, data.batch)
            graph_reprs.append(graph_repr)

        # Concatenate all layer representations
        combined = torch.cat(graph_reprs, dim=1)

        # Final prediction
        out = self.output_mlp(combined)
        out = self.final_activation(out)

        return out

    def __repr__(self):
        return (
            f"PlanE(\n"
            f"  node_features={self.num_node_features},\n"
            f"  edge_features={self.num_edge_features},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  num_classes={self.num_classes},\n"
            f"  num_layers={self.num_layers},\n"
            f"  dropout={self.dropout},\n"
            f"  task={self.task}\n"
            f")"
        )


class SimplePlanE(nn.Module):
    """
    Ultra-simple PlanE model with minimal configuration.

    Just specify input/output dimensions and you're good to go!
    Uses sensible defaults for everything else.

    Args:
        num_node_features (int): Number of input node features
        num_classes (int): Number of output classes
        categorical_node_features (bool): If True, use Embedding for node features.
                                          If False, use Linear. Default: False

    Example:
        >>> model = SimplePlanE(num_node_features=1, num_classes=4)
        >>> output = model(data)
    """

    def __init__(self, num_node_features, num_classes, categorical_node_features=False):
        super().__init__()
        self.model = PlanE(
            num_node_features=num_node_features,
            num_classes=num_classes,
            hidden_dim=64,
            num_layers=3,
            dropout=0.1,
            categorical_node_features=categorical_node_features,
            use_neighbors=True,
            use_triconnected=True,
            use_biconnected=True,
            use_global_readout=True,
            positional_encoding_dim=16,
            task="classification",
        )

    def forward(self, data):
        return self.model(data)

    def __repr__(self):
        return f"SimplePlanE(model={self.model})"
