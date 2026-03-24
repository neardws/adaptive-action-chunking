"""
Train the k-Selector MLP on oracle-labeled data.

Usage:
  python scripts/train_selector.py \
    --features_dir data/features \
    --labels data/oracle_labels.jsonl \
    --output checkpoints/selector.pt
"""

import sys, argparse, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.selector.model import KSelectorMLP, IDX_TO_K
from src.selector.dataset import OracleLabelDataset


def train(args):
    device = torch.device(args.device)

    # Dataset
    dataset = OracleLabelDataset(args.features_dir, args.labels)
    n_val = max(1, int(len(dataset) * 0.15))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Model
    model = KSelectorMLP(
        feature_dim=args.feature_dim,
        state_dim=args.state_dim,
        hidden_dims=args.hidden_dims,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    history = []

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss, train_correct = 0.0, 0
        for feat, state, label in train_loader:
            feat, state, label = feat.to(device), state.to(device), label.to(device)
            logits = model(feat, state)
            loss = criterion(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(label)
            train_correct += (logits.argmax(1) == label).sum().item()
        scheduler.step()

        # Val
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for feat, state, label in val_loader:
                feat, state, label = feat.to(device), state.to(device), label.to(device)
                logits = model(feat, state)
                val_correct += (logits.argmax(1) == label).sum().item()

        train_acc = train_correct / n_train
        val_acc = val_correct / n_val
        history.append({"epoch": epoch, "train_acc": train_acc, "val_acc": val_acc})

        if (epoch + 1) % 10 == 0:
            print(f"[{epoch+1:3d}/{args.epochs}] train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": vars(args)}, args.output)

    print(f"\nBest val_acc: {best_val_acc:.3f}")
    print(f"Saved to: {args.output}")
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--labels", default="data/oracle_labels.jsonl")
    parser.add_argument("--output", default="checkpoints/selector.pt")
    parser.add_argument("--feature_dim", type=int, default=1152)
    parser.add_argument("--state_dim", type=int, default=32)
    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128, 64])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
