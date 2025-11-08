"""
Basic tests for mini-PlanE
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from   plane                    import PlanE, SimplePlanE
import torch
from   torch_geometric.data     import Data


def create_simple_graph():
    """Create a simple planar graph for testing (K4 - complete graph on 4 vertices)."""
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 0, 0, 2, 1, 3],
            [1, 0, 2, 1, 3, 2, 0, 3, 2, 0, 3, 1],
        ],
        dtype=torch.long,
    )

    data = Data(
        x=torch.ones((4, 1)),  # 4 nodes, 1 feature each
        edge_index=edge_index,
        y=torch.tensor([0]),  # Class label
        batch=torch.zeros(4, dtype=torch.long),  # All nodes in same graph
    )

    # Add minimal SPQR preprocessing (placeholder)
    num_nodes = 4
    num_edges = edge_index.size(1)

    data.spqr_batch = torch.zeros(1, dtype=torch.long)
    data.spqr_type = torch.zeros(1, dtype=torch.long)
    data.spqr_order = torch.zeros(1, dtype=torch.long)

    data.spqr_read_from_e = torch.stack(
        [
            torch.zeros(num_edges, dtype=torch.long),
            edge_index[0],
            edge_index[1],
            torch.arange(num_edges, dtype=torch.long),
            torch.zeros(num_edges, dtype=torch.long),
            torch.zeros(num_edges, dtype=torch.long),
        ]
    )

    data.spqr_edge_index = torch.zeros((2, 0), dtype=torch.long)
    data.spqr_edge_attr = torch.zeros(0, dtype=torch.long)

    data.b_batch = torch.zeros(1, dtype=torch.long)
    data.b_order = torch.zeros(1, dtype=torch.long)
    data.c_order = torch.zeros(num_nodes, dtype=torch.long)

    data.g_read_from_spqr = torch.zeros((2, num_nodes), dtype=torch.long)
    data.g_read_from_spqr[1] = torch.arange(num_nodes)

    data.g_read_from_b = torch.zeros((2, num_nodes), dtype=torch.long)
    data.g_read_from_b[1] = torch.arange(num_nodes)

    data.b_read_from_spqr_root = torch.zeros((2, 1), dtype=torch.long)

    data.bc_edge_index = torch.zeros((2, 0), dtype=torch.long)
    data.cb_edge_index = torch.zeros((2, 0), dtype=torch.long)

    return data


def test_simple_plane():
    """Test SimplePlanE model."""
    print("Testing SimplePlanE...")

    model = SimplePlanE(num_node_features=1, num_classes=4)
    data = create_simple_graph()

    # Forward pass
    output = model(data)

    assert output.shape == (1, 4), f"Expected shape (1, 4), got {output.shape}"
    print(f"  * SimplePlanE forward pass successful: {output.shape}")


def test_plane():
    """Test PlanE model with custom configuration."""
    print("\nTesting PlanE...")

    model = PlanE(
        num_node_features=1,
        num_classes=4,
        hidden_dim=32,
        num_layers=2,
        dropout=0.0,
        use_neighbors=True,
        use_triconnected=True,
        use_biconnected=True,
        use_global_readout=True,
    )

    data = create_simple_graph()

    # Forward pass
    output = model(data)

    assert output.shape == (1, 4), f"Expected shape (1, 4), got {output.shape}"
    print(f"  * PlanE forward pass successful: {output.shape}")


def test_plane_minimal():
    """Test minimal PlanE (only neighbors)."""
    print("\nTesting minimal PlanE (neighbors only)...")

    model = PlanE(
        num_node_features=1,
        num_classes=4,
        hidden_dim=32,
        num_layers=2,
        use_neighbors=True,
        use_triconnected=False,
        use_biconnected=False,
        use_global_readout=False,
    )

    data = create_simple_graph()

    # Forward pass
    output = model(data)

    assert output.shape == (1, 4), f"Expected shape (1, 4), got {output.shape}"
    print(f"  * Minimal PlanE forward pass successful: {output.shape}")


def test_backward():
    """Test backward pass."""
    print("\nTesting backward pass...")

    model = SimplePlanE(num_node_features=1, num_classes=4)
    data = create_simple_graph()

    # Forward pass
    output = model(data)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check gradients exist
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No gradients computed!"

    print(f"  * Backward pass successful")


def test_repr():
    """Test model __repr__."""
    print("\nTesting model representations...")

    model1 = SimplePlanE(num_node_features=1, num_classes=4)
    repr1 = repr(model1)
    assert "SimplePlanE" in repr1
    print(f"  * SimplePlanE repr: {repr1[:50]}...")

    model2 = PlanE(num_node_features=1, num_classes=4)
    repr2 = repr(model2)
    assert "PlanE" in repr2
    print(f"  * PlanE repr: {repr2[:50]}...")


if __name__ == "__main__":
    print("=" * 60)
    print("Running PlanE Minimal Tests")
    print("=" * 60)

    try:
        test_simple_plane()
        test_plane()
        test_plane_minimal()
        test_backward()
        test_repr()

        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
