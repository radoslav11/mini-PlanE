"""
Complete example: Training PlanE on Genus Graph Classification

This example shows how to use PlanE Minimal with the GenusGraph dataset
for classifying graphs by their genus (topological property).

Usage:
    python genus_classification.py --dataset hard --epochs 50
"""

import argparse
import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from   plane                    import PlanE
import torch
from   torch                    import nn
from   torch_geometric.loader   import DataLoader

# Import from GenusGNN project
from   GenusGNN.data_utils      import json_to_pyg, load_json_dataset


def load_genus_dataset(dataset_name="hard"):
    """
    Load GenusGraph dataset.

    Args:
        dataset_name: 'easy', 'hard', or 'expert'

    Returns:
        train_data, test_data: Lists of PyG Data objects
    """
    # Map dataset name to directory
    data_dirs = {
        "easy": os.path.join("..", "..", "data"),
        "hard": os.path.join("..", "..", "data_hard"),
        "expert": os.path.join("..", "..", "data_expert"),
    }

    if dataset_name not in data_dirs:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    data_dir = data_dirs[dataset_name]

    # Load graphs
    print(f"Loading {dataset_name} dataset from {data_dir}")
    graphs = load_json_dataset(data_dir)

    # Convert to PyG format
    data_list = [json_to_pyg(g, task="genus") for g in graphs]

    # Split train/test (70/30)
    num_train = int(0.7 * len(data_list))
    train_data = data_list[:num_train]
    test_data = data_list[num_train:]

    print(
        f"Loaded {len(train_data)} training graphs, {len(test_data)} test graphs"
    )

    return train_data, test_data


def add_genus_preprocessing(data):
    """
    Add simplified SPQR preprocessing for genus graphs.

    NOTE: This uses the placeholder preprocessing. For better results,
    use full SPQR preprocessing from original PlanE.
    """
    from preprocess_data import add_spqr_preprocessing_placeholder

    return add_spqr_preprocessing_placeholder(data)


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        out = model(batch)
        loss = criterion(out, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)

        out = model(batch)
        loss = criterion(out, batch.y)

        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs

    return total_loss / total, correct / total


def main(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load dataset
    train_dataset, test_dataset = load_genus_dataset(args.dataset)

    # Add SPQR preprocessing
    print("Adding SPQR preprocessing...")
    train_dataset = [add_genus_preprocessing(d) for d in train_dataset]
    test_dataset = [add_genus_preprocessing(d) for d in test_dataset]

    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # Model
    print(f"\nInitializing PlanE model...")
    model = PlanE(
        num_node_features=1,  # GenusGraph uses degree as feature
        num_classes=4,  # G0, G1, G2, G3
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_neighbors=True,
        use_triconnected=True,
        use_biconnected=True,
        use_global_readout=True,
        positional_encoding_dim=16,
        task="classification",
    )
    model = model.to(device)
    print(model)

    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # Training loop
    print(f"\n{'='*60}")
    print(f"Training on GenusGraph-{args.dataset.upper()} dataset")
    print(f"{'='*60}\n")

    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Print progress
        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc*100:.2f}%",
            end="",
        )

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            print(f" * NEW BEST!")
            if args.save_model:
                torch.save(model.state_dict(), args.save_model)
        else:
            print()

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best test accuracy: {best_test_acc*100:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PlanE on Genus Graph Classification"
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="hard",
        choices=["easy", "hard", "expert"],
        help="Dataset difficulty",
    )

    # Model
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Hidden dimension"
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="Number of PlanE layers"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability"
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Weight decay"
    )

    # Misc
    parser.add_argument(
        "--save_model", type=str, default="", help="Path to save best model"
    )

    args = parser.parse_args()
    main(args)
