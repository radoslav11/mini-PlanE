# mini-PlanE

**A simplified, easy-to-use interface for PlanE (Representation Learning over Planar Graphs)**

Perfect for users new to planar graph neural networks! No complex configuration flags needed - just plug and play.

---

## What is PlanE?

PlanE is a powerful graph neural network designed specifically for **planar graphs** - graphs that can be drawn on a plane without edge crossings. PlanE learns **complete invariants** while remaining practically scalable, inspired by the classical Hopcroft-Tarjan planar graph isomorphism algorithm.

**Key advantages:**
- **More expressive** than standard GNNs (GCN, GIN, GAT)
- **Captures planar structure** through SPQR tree decomposition
- **Scalable** for real-world graphs
- **Easy to use** with sensible defaults

---

## Cite

If you make use of this code, or its accompanying [paper](https://arxiv.org/abs/2307.01180), please cite this work as follows:

```bibtex
@inproceedings{DimitrovZAC23,
  author    = {Radoslav Dimitrov and Zeyang Zhao and
               Ralph Abboud and
               {\.I}smail {\.I}lkan Ceylan},
  title     = {PlanE: Representation Learning over Planar Graphs},
  booktitle = {Proceedings of the Thirty-Seventh Annual Conference on
               Advances in Neural Information Processing Systems, {NeurIPS}},
  year      = {2023}
}
```

**Paper:** [https://arxiv.org/abs/2307.01180](https://arxiv.org/abs/2307.01180)
**Original PlanE Repository:** [https://github.com/ZZYSonny/PlanE](https://github.com/ZZYSonny/PlanE)

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mini-PlanE.git
cd mini-PlanE

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### Basic Usage

```python
from plane import PlanE, SimplePlanE
import torch
from torch_geometric.data import Data

# Option 1: Ultra-simple (just specify input/output)
model = SimplePlanE(
    num_node_features=1,
    num_classes=4
)

# Option 2: More control with sensible defaults
model = PlanE(
    num_node_features=1,
    num_classes=4,
    hidden_dim=64,      # Hidden dimension
    num_layers=3,       # Number of layers
    dropout=0.1         # Dropout rate
)

# Forward pass (data must have SPQR preprocessing - see below)
output = model(data)
```

---

## Training Example

```python
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from plane import SimplePlanE

# 1. Prepare your dataset
# NOTE: Graphs must be preprocessed with SPQR decomposition
# See examples/preprocess_data.py for details
from examples.preprocess_data import preprocess_planar_graphs

# Load your planar graphs
graphs = load_your_planar_graphs()  # List of PyG Data objects
preprocessed_graphs = preprocess_planar_graphs(graphs)

# 2. Create data loaders
train_loader = DataLoader(preprocessed_graphs, batch_size=32, shuffle=True)

# 3. Initialize model
model = SimplePlanE(num_node_features=1, num_classes=4)

# 4. Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(100):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f'Epoch {epoch}: Loss = {total_loss / len(train_loader):.4f}')
```

---

## Understanding the Model

### Architecture Options

PlanE aggregates information from multiple structural levels:

1. **Neighbors** (`use_neighbors=True`): Like standard GNNs, aggregate from 1-hop neighbors
2. **Triconnected components** (`use_triconnected=True`): Capture fundamental planar structure
3. **Biconnected components** (`use_biconnected=True`): Capture cut vertices and bridges
4. **Global readout** (`use_global_readout=True`): Graph-level information

You can enable/disable any combination:

```python
# Minimal model (just neighbors - equivalent to GNN)
model = PlanE(
    num_node_features=1,
    num_classes=4,
    use_neighbors=True,
    use_triconnected=False,
    use_biconnected=False,
    use_global_readout=False
)

# Full model (all structural levels - maximum expressivity)
model = PlanE(
    num_node_features=1,
    num_classes=4,
    use_neighbors=True,
    use_triconnected=True,
    use_biconnected=True,
    use_global_readout=True
)
```

### Model Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_node_features` | - | **Required.** Number of node features (or embedding size) |
| `num_edge_features` | 0 | Number of edge features (0 if no edge features) |
| `hidden_dim` | 64 | Hidden dimension for all layers |
| `num_classes` | 2 | Number of output classes |
| `num_layers` | 3 | Number of PlanE layers |
| `dropout` | 0.0 | Dropout probability |
| `use_neighbors` | True | Aggregate from 1-hop neighbors |
| `use_triconnected` | True | Aggregate from triconnected components |
| `use_biconnected` | True | Aggregate from biconnected components |
| `use_global_readout` | True | Use global graph readout |
| `positional_encoding_dim` | 16 | Dimension for positional encodings |
| `task` | 'classification' | 'classification' or 'regression' |

---

## Data Preprocessing

**Important:** PlanE requires planar graphs to be preprocessed with SPQR tree decomposition.

### Option 1: Use our preprocessing script

```python
from examples.preprocess_data import preprocess_planar_graphs
from torch_geometric.data import Data

# Your planar graphs (as PyG Data objects)
graphs = [
    Data(x=..., edge_index=..., y=...),
    Data(x=..., edge_index=..., y=...),
    ...
]

# Preprocess (adds SPQR attributes)
preprocessed = preprocess_planar_graphs(graphs)

# Now ready for PlanE!
model = SimplePlanE(num_node_features=1, num_classes=4)
output = model(preprocessed[0])
```

### Option 2: Use the full PlanE preprocessing

If you need the full preprocessing from the original PlanE repository:

```bash
# Clone the original PlanE repo
git clone https://github.com/ZZYSonny/PlanE.git

# Use their preprocessing
python -m preprocess.prepare --dataset your_dataset
```

---

## Examples

We provide several complete examples in the `examples/` directory:

- **`examples/train_simple.py`** - Simple training script for planar graph classification
- **`examples/preprocess_data.py`** - Data preprocessing utilities
- **`examples/quickstart.ipynb`** - Interactive Jupyter notebook tutorial
- **`examples/genus_classification.py`** - Example: classifying graphs by genus
- **`examples/custom_dataset.py`** - How to use PlanE with your own dataset

Run an example:

```bash
python examples/train_simple.py --dataset genus_hard --epochs 50
```

---

## Frequently Asked Questions

### Q: What if my graphs aren't planar?

PlanE is specifically designed for planar graphs. For non-planar graphs, consider:
- Standard GNNs (GCN, GIN, GAT)
- More expressive architectures like PPGN or subgraph GNNs
- Graph transformers

### Q: How do I know if my graph is planar?

```python
import networkx as nx

G = nx.Graph()
# ... add edges ...

is_planar = nx.check_planarity(G)[0]
print(f"Graph is planar: {is_planar}")
```

### Q: Do I need edge features?

No! PlanE works great with just node features. If you have edge features, set `num_edge_features` appropriately.

### Q: What datasets work well with PlanE?

PlanE excels on:
- Molecular graphs (many are planar or nearly planar)
- Road networks
- Circuit graphs
- Geographic networks
- Genus classification tasks

### Q: How does PlanE compare to standard GNNs?

On planar graph tasks, PlanE typically achieves:
- **Higher expressivity** (can distinguish more graph structures)
- **Better accuracy** on structure-sensitive tasks
- **Comparable speed** to 3-4 layer GNNs

---

## Troubleshooting

### Error: "Data object missing SPQR attributes"

**Solution:** Your graphs need SPQR preprocessing. Use `preprocess_planar_graphs()`:

```python
from examples.preprocess_data import preprocess_planar_graphs
preprocessed = preprocess_planar_graphs(graphs)
```

### Error: "Graph is not planar"

**Solution:** PlanE only works with planar graphs. Check planarity:

```python
import networkx as nx
G = nx.from_edgelist(edge_list)
is_planar, _ = nx.check_planarity(G)
```

### Error: "CUDA out of memory"

**Solution:** Reduce batch size or hidden dimension:

```python
model = PlanE(hidden_dim=32, ...)  # Smaller hidden dim
loader = DataLoader(dataset, batch_size=16, ...)  # Smaller batch
```

---

## Comparison with Original PlanE

**PlanE Minimal** vs **Original PlanE**:

| Feature | PlanE Minimal | Original PlanE |
|---------|---------------|----------------|
| Configuration | Simple, intuitive | Complex flags like `flags_plane_agg="n_t_b"` |
| Documentation | Extensive examples | Research-focused |
| Ease of use | Plug-and-play | Requires understanding of internals |
| Flexibility | Sensible defaults | Highly configurable |
| Best for | Quick prototyping, new users | Research, fine-tuning |

**When to use PlanE Minimal:**
- You're new to planar GNNs
- You want quick results without configuration
- You're prototyping or teaching

**When to use Original PlanE:**
- You need fine-grained control over every component
- You're reproducing research results
- You're pushing performance limits

---

## License

MIT License - see LICENSE file for details.

---

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

---

## Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/mini-PlanE/issues)
- **Questions:** [GitHub Discussions](https://github.com/yourusername/mini-PlanE/discussions)
- **Original PlanE:** [https://github.com/ZZYSonny/PlanE](https://github.com/ZZYSonny/PlanE)

---

**Happy learning on planar graphs!**
