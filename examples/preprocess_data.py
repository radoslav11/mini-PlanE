"""
Data preprocessing utilities for PlanE.

This module provides functions to add SPQR tree decomposition to planar graphs,
which is required for PlanE to work.

NOTE: This is a simplified preprocessing interface. For production use,
consider using the full preprocessing from the original PlanE repository.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import networkx as nx
import torch
from   torch_geometric.data     import Data


def check_planarity(edge_index, num_nodes):
    """
    Check if a graph is planar.

    Args:
        edge_index: Edge index tensor [2, num_edges]
        num_nodes: Number of nodes

    Returns:
        bool: True if graph is planar
    """
    # Convert to NetworkX
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().tolist()
    G.add_edges_from(edges)

    is_planar, _ = nx.check_planarity(G)
    return is_planar


def add_spqr_preprocessing_placeholder(data):
    """
    Placeholder for SPQR preprocessing.

    NOTE: This is a simplified placeholder. For production use, you should use
    the full SPQR preprocessing from the original PlanE repository, which:
    - Computes SPQR tree decomposition
    - Adds triconnected and biconnected component information
    - Adds canonical orderings for planarity

    For now, this adds dummy attributes so the model can run, but you won't
    get the full benefits of PlanE's planar-aware architecture.

    Args:
        data: PyTorch Geometric Data object

    Returns:
        Data: Data object with SPQR attributes added
    """
    num_nodes = data.x.size(0) if data.x is not None else data.num_nodes

    # Add dummy SPQR attributes
    # In a real implementation, these would be computed from SPQR decomposition

    # Dummy values - replace with real SPQR preprocessing
    data.spqr_batch = torch.zeros(1, dtype=torch.long)  # 1 component
    data.spqr_type = torch.zeros(1, dtype=torch.long)  # Type 0 (S)
    data.spqr_order = torch.zeros(1, dtype=torch.long)

    # Edge information for triconnected components
    # Format: [component_id, node_u, node_v, edge_id, code1, code2]
    num_edges = data.edge_index.size(1)
    data.spqr_read_from_e = torch.stack(
        [
            torch.zeros(num_edges, dtype=torch.long),  # component_id
            data.edge_index[0],  # node_u
            data.edge_index[1],  # node_v
            torch.arange(num_edges, dtype=torch.long),  # edge_id
            torch.zeros(num_edges, dtype=torch.long),  # code1
            torch.zeros(num_edges, dtype=torch.long),  # code2
        ]
    )

    # SPQR tree structure
    data.spqr_edge_index = torch.zeros((2, 0), dtype=torch.long)
    data.spqr_edge_attr = torch.zeros(0, dtype=torch.long)

    # Biconnected components
    data.b_batch = torch.zeros(1, dtype=torch.long)
    data.b_order = torch.zeros(1, dtype=torch.long)
    data.c_order = torch.zeros(num_nodes, dtype=torch.long)

    # Read indices
    data.g_read_from_spqr = torch.zeros((2, num_nodes), dtype=torch.long)
    data.g_read_from_spqr[1] = torch.arange(num_nodes)

    data.g_read_from_b = torch.zeros((2, num_nodes), dtype=torch.long)
    data.g_read_from_b[1] = torch.arange(num_nodes)

    data.b_read_from_spqr_root = torch.zeros((2, 1), dtype=torch.long)

    # Block-cutpoint graph
    data.bc_edge_index = torch.zeros((2, 0), dtype=torch.long)
    data.cb_edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Batch attribute (all nodes belong to graph 0 for single graph)
    data.batch = torch.zeros(num_nodes, dtype=torch.long)

    return data


def preprocess_planar_graph(data, check_planar=True):
    """
    Preprocess a single planar graph for PlanE.

    Args:
        data: PyTorch Geometric Data object
        check_planar: Whether to check if graph is planar (default: True)

    Returns:
        Data: Preprocessed graph with SPQR attributes

    Raises:
        ValueError: If graph is not planar and check_planar=True
    """
    # Check planarity
    if check_planar:
        num_nodes = data.x.size(0) if data.x is not None else data.num_nodes
        if not check_planarity(data.edge_index, num_nodes):
            raise ValueError("Graph is not planar!")

    # Add SPQR preprocessing
    # NOTE: This is a placeholder - use full PlanE preprocessing for production
    data = add_spqr_preprocessing_placeholder(data)

    return data


def preprocess_planar_graphs(graphs, check_planar=True, verbose=True):
    """
    Preprocess a list of planar graphs for PlanE.

    Args:
        graphs: List of PyTorch Geometric Data objects
        check_planar: Whether to check if graphs are planar (default: True)
        verbose: Whether to print progress (default: True)

    Returns:
        List[Data]: Preprocessed graphs with SPQR attributes
    """
    preprocessed = []
    skipped = 0

    for i, graph in enumerate(graphs):
        try:
            preprocessed_graph = preprocess_planar_graph(graph, check_planar)
            preprocessed.append(preprocessed_graph)
        except ValueError as e:
            if verbose:
                print(f"Skipping graph {i}: {e}")
            skipped += 1

    if verbose:
        print(f"\nPreprocessed {len(preprocessed)} graphs ({skipped} skipped)")

    return preprocessed


def use_full_plane_preprocessing():
    """
    Instructions for using the full PlanE preprocessing.

    The placeholder preprocessing above won't give you the full benefits of PlanE.
    For production use, follow these steps:
    """
    instructions = """
    To use full SPQR preprocessing:

    1. Clone the original PlanE repository:
       git clone https://github.com/ZZYSonny/PlanE.git

    2. Install dependencies:
       cd PlanE
       conda env create -f environment.yml
       conda activate plane

    3. Add your dataset to PlanE/datasets/

    4. Run preprocessing:
       python -m preprocess.prepare --dataset your_dataset

    5. The preprocessed data will include full SPQR decomposition with:
       - Triconnected components (S, P, R types)
       - Biconnected components
       - Cut vertices
       - Canonical orderings
       - SPQR tree structure

    6. Copy the preprocessed data back to use with PlanE Minimal

    Alternatively, you can use the PlanE preprocessing code directly:
       from PlanE.preprocess.data_process import process
       preprocessed_data = process(your_data)
    """
    print(instructions)


if __name__ == "__main__":
    # Example usage
    print("PlanE Preprocessing Utilities")
    print("=" * 50)

    # Create a simple planar graph
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long
    )

    data = Data(
        x=torch.ones((4, 1)), edge_index=edge_index, y=torch.tensor([0])
    )

    print("\nExample: Preprocessing a simple planar graph (K4)")
    print(f"Nodes: {data.x.size(0)}")
    print(f"Edges: {data.edge_index.size(1)}")

    # Check planarity
    is_planar = check_planarity(data.edge_index, data.x.size(0))
    print(f"Is planar: {is_planar}")

    # Preprocess
    preprocessed = preprocess_planar_graph(data)
    print(f"\nPreprocessed attributes added:")
    for key in preprocessed.keys():
        if (
            key.startswith("spqr")
            or key.startswith("b_")
            or key.startswith("c_")
            or key.startswith("g_")
        ):
            print(f"  - {key}")

    print("\n" + "=" * 50)
    print("NOTE: This is placeholder preprocessing!")
    print("For production use, see instructions below:\n")
    use_full_plane_preprocessing()
