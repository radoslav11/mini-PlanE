"""
Simple training script for PlanE on planar graph classification.

Usage:
    python train_simple.py --dataset genus_hard --epochs 50
    python train_simple.py --dataset your_dataset --hidden_dim 128 --num_layers 4
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(batch)
        loss = criterion(out, batch.y)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.num_graphs

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
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
    print(f"Using device: {device}")

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    # TODO: Replace with your dataset loading
    # For now, this is a placeholder
    print("ERROR: Please implement dataset loading for your specific dataset")
    print("See examples/genus_classification.py for a complete example")
    return

    # Example dataset loading (uncomment and modify for your dataset):
    # from your_dataset_module import load_dataset
    # train_dataset, test_dataset = load_dataset(args.dataset)
    #
    # train_loader = DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=True
    # )
    # test_loader = DataLoader(
    #     test_dataset, batch_size=args.batch_size, shuffle=False
    # )
    #
    # # Model
    # print(f"\nInitializing model...")
    # if args.simple:
    #     # Ultra-simple model
    #     model = SimplePlanE(
    #         num_node_features=args.num_node_features,
    #         num_classes=args.num_classes,
    #     )
    # else:
    #     # Customizable model
    #     model = PlanE(
    #         num_node_features=args.num_node_features,
    #         num_edge_features=args.num_edge_features,
    #         hidden_dim=args.hidden_dim,
    #         num_classes=args.num_classes,
    #         num_layers=args.num_layers,
    #         dropout=args.dropout,
    #         use_neighbors=args.use_neighbors,
    #         use_triconnected=args.use_triconnected,
    #         use_biconnected=args.use_biconnected,
    #         use_global_readout=args.use_global_readout,
    #         positional_encoding_dim=args.pe_dim,
    #         task="classification",
    #     )
    #
    # model = model.to(device)
    # print(model)
    #
    # # Optimizer and loss
    # optimizer = torch.optim.Adam(
    #     model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    # )
    # criterion = nn.CrossEntropyLoss()
    #
    # # Training loop
    # print(f"\nTraining for {args.epochs} epochs...\n")
    # best_test_acc = 0.0
    #
    # for epoch in range(1, args.epochs + 1):
    #     # Train
    #     train_loss, train_acc = train_epoch(
    #         model, train_loader, optimizer, criterion, device
    #     )
    #
    #     # Evaluate
    #     test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    #
    #     # Print progress
    #     if epoch % args.log_interval == 0:
    #         print(
    #             f"Epoch {epoch:3d} | "
    #             f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
    #             f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
    #         )
    #
    #     # Save best model
    #     if test_acc > best_test_acc:
    #         best_test_acc = test_acc
    #         if args.save_model:
    #             torch.save(model.state_dict(), args.save_model)
    #             print(f"    Saved best model (acc={best_test_acc:.4f})")
    #
    # print(f"\nTraining complete!")
    # print(f"Best test accuracy: {best_test_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PlanE on planar graph classification"
    )

    # Dataset
    parser.add_argument(
        "--dataset", type=str, required=True, help="Dataset name"
    )
    parser.add_argument(
        "--num_node_features",
        type=int,
        default=1,
        help="Number of node features",
    )
    parser.add_argument(
        "--num_edge_features",
        type=int,
        default=0,
        help="Number of edge features",
    )
    parser.add_argument(
        "--num_classes", type=int, default=4, help="Number of output classes"
    )

    # Model
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use SimplePlanE (ignores other model args)",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Hidden dimension"
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="Number of PlanE layers"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout probability"
    )
    parser.add_argument(
        "--pe_dim", type=int, default=16, help="Positional encoding dimension"
    )

    # Architecture options
    parser.add_argument(
        "--use_neighbors",
        type=bool,
        default=True,
        help="Aggregate from neighbors",
    )
    parser.add_argument(
        "--use_triconnected",
        type=bool,
        default=True,
        help="Aggregate from triconnected components",
    )
    parser.add_argument(
        "--use_biconnected",
        type=bool,
        default=True,
        help="Aggregate from biconnected components",
    )
    parser.add_argument(
        "--use_global_readout",
        type=bool,
        default=True,
        help="Use global readout",
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

    # Logging
    parser.add_argument(
        "--log_interval", type=int, default=10, help="Log every N epochs"
    )
    parser.add_argument(
        "--save_model", type=str, default="", help="Path to save best model"
    )

    args = parser.parse_args()
    main(args)
