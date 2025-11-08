# PlanE Minimal - 5 Minute Quick Start

Get started with PlanE in just 5 minutes!

## Installation

```bash
cd mini-PlanE
pip install -r requirements.txt
```

## Simplest Possible Usage

```python
from plane import SimplePlanE

# 1. Create model (just specify input/output!)
model = SimplePlanE(num_node_features=1, num_classes=4)

# 2. Forward pass (data must have SPQR preprocessing)
output = model(data)

# That's it!
```

## With More Control

```python
from plane import PlanE

model = PlanE(
    num_node_features=1,
    num_classes=4,
    hidden_dim=64,      # Hidden dimension
    num_layers=3,       # Number of layers
    dropout=0.1         # Dropout rate
)
```

## Complete Training Example

```python
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from plane import SimplePlanE

# Load your preprocessed planar graphs
# (See examples/preprocess_data.py for preprocessing)
train_loader = DataLoader(train_graphs, batch_size=32)

# Create model
model = SimplePlanE(num_node_features=1, num_classes=4)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
```

## Architecture Options

Control what PlanE aggregates from:

```python
model = PlanE(
    num_node_features=1,
    num_classes=4,

    # Choose what to aggregate from:
    use_neighbors=True,       # 1-hop neighbors (like GNN)
    use_triconnected=True,    # Triconnected components (planar structure)
    use_biconnected=True,     # Biconnected components
    use_global_readout=True   # Global graph information
)
```

**Tip:** Start with all `True` for maximum expressivity!

## Data Preprocessing

PlanE needs SPQR decomposition. Quick option:

```python
from examples.preprocess_data import preprocess_planar_graphs

# Your planar graphs
graphs = [...]  # List of PyG Data objects

# Add SPQR preprocessing
preprocessed = preprocess_planar_graphs(graphs)

# Ready for PlanE!
model = SimplePlanE(num_node_features=1, num_classes=4)
output = model(preprocessed[0])
```

**Note:** This uses placeholder preprocessing. For production, use full SPQR preprocessing from original PlanE repository.

## Examples

Run complete examples:

```bash
# Genus classification example
python examples/genus_classification.py --dataset hard --epochs 50

# Run tests
python tests/test_basic.py

# Interactive tutorial
jupyter notebook examples/quickstart.ipynb
```
