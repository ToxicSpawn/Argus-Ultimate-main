"""DeepLOB Training Pipeline — Push 40.

Trains a 3-class MLP (short / flat / long) on LOB snapshot data and
exports a TorchScript model to models/deeplob_weights.pt for use by
DeepLOBLiveBridge.

Data format
-----------
CSV with 40 feature columns (bid_p0..bid_p9, ask_p0..ask_p9,
bid_v0..bid_v9, ask_v0..ask_v9) + 1 label column (0=short, 1=flat, 2=long).

If data/lob_snapshots.csv is absent, synthetic data is generated.

Usage
-----
    python scripts/train_deeplob.py
    python scripts/train_deeplob.py --epochs 100 --lr 0.0005 --hidden 128
    python scripts/train_deeplob.py --data data/my_lob.csv --out models/deeplob_v2.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
)
logger = logging.getLogger("train_deeplob")

_DEFAULT_DATA   = os.path.join("data", "lob_snapshots.csv")
_DEFAULT_OUT    = os.path.join("models", "deeplob_weights.pt")
_N_FEATURES     = 40
_N_CLASSES      = 3
_SYNTHETIC_N    = 20_000


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_or_generate(path: str, n_synthetic: int = _SYNTHETIC_N):
    """Load CSV or generate synthetic LOB data.

    Returns
    -------
    X : np.ndarray  shape (N, 40)  float32
    y : np.ndarray  shape (N,)     int64
    """
    if os.path.exists(path):
        logger.info("Loading LOB data from %s", path)
        import csv
        rows, labels = [], []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                feat = [float(row[f"feat_{i}"]) for i in range(_N_FEATURES)]
                rows.append(feat)
                labels.append(int(row["label"]))
        X = np.array(rows, dtype=np.float32)
        y = np.array(labels, dtype=np.int64)
        logger.info("Loaded %d samples from CSV", len(X))
    else:
        logger.info("No CSV found at %s — generating %d synthetic samples", path, n_synthetic)
        rng = np.random.default_rng(seed=42)

        # Synthetic: bid prices slightly below 0, ask slightly above 0 (normalised by mid)
        bid_prices = rng.uniform(-0.002, 0.000, (n_synthetic, 10)).astype(np.float32)
        ask_prices = rng.uniform(0.000, 0.002,  (n_synthetic, 10)).astype(np.float32)
        bid_vols   = rng.exponential(2.0, (n_synthetic, 10)).astype(np.float32)
        ask_vols   = rng.exponential(2.0, (n_synthetic, 10)).astype(np.float32)
        X = np.concatenate([bid_prices, ask_prices, np.log1p(bid_vols), np.log1p(ask_vols)], axis=1)

        # Label heuristic: bid depth > ask depth -> long, else short, else flat
        bid_depth = bid_vols.sum(axis=1)
        ask_depth = ask_vols.sum(axis=1)
        ratio = bid_depth / (ask_depth + 1e-8)
        y = np.where(ratio > 1.15, 2,   # long
            np.where(ratio < 0.87, 0,   # short
            1)).astype(np.int64)         # flat

        logger.info("Synthetic label distribution: short=%d flat=%d long=%d",
                    (y == 0).sum(), (y == 1).sum(), (y == 2).sum())

    return X, y


def train_val_split(X, y, val_frac: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_val = int(len(X) * val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(n_features: int, n_hidden: int, n_classes: int):
    """Build a 3-layer MLP using PyTorch."""
    import torch
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(n_features, n_hidden),
        nn.ReLU(),
        nn.BatchNorm1d(n_hidden),
        nn.Dropout(0.3),
        nn.Linear(n_hidden, n_hidden // 2),
        nn.ReLU(),
        nn.BatchNorm1d(n_hidden // 2),
        nn.Dropout(0.2),
        nn.Linear(n_hidden // 2, n_classes),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X_train, y_train, X_val, y_val,
    n_hidden: int = 64,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 0.001,
    patience: int = 8,
    out_path: str = _DEFAULT_OUT,
) -> None:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s | train=%d val=%d features=%d hidden=%d",
                device, len(X_train), len(X_val), _N_FEATURES, n_hidden)

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_vl = torch.tensor(X_val,   dtype=torch.float32).to(device)
    y_vl = torch.tensor(y_val,   dtype=torch.long).to(device)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )

    model     = build_model(_N_FEATURES, n_hidden, _N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=4, factor=0.5, verbose=True,
    )

    best_val_acc = 0.0
    best_state   = None
    no_improve   = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimiser.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_vl).argmax(dim=1)
            val_acc   = (val_preds == y_vl).float().mean().item()

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)
        logger.info("Epoch %3d/%d | loss=%.4f val_acc=%.4f", epoch, epochs, avg_loss, val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    # Restore best weights and export TorchScript
    if best_state:
        model.load_state_dict(best_state)
    model.eval().to("cpu")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scripted = torch.jit.script(model)
    scripted.save(out_path)
    logger.info("TorchScript model saved to %s (best_val_acc=%.4f)", out_path, best_val_acc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeepLOB MLP Training Pipeline (Push 40)")
    p.add_argument("--data",       default=_DEFAULT_DATA,  help="Path to LOB CSV")
    p.add_argument("--out",        default=_DEFAULT_OUT,   help="Output .pt path")
    p.add_argument("--epochs",     default=50,  type=int)
    p.add_argument("--lr",         default=0.001, type=float)
    p.add_argument("--hidden",     default=64,  type=int)
    p.add_argument("--batch-size", default=256, type=int)
    p.add_argument("--patience",   default=8,   type=int)
    p.add_argument("--val-frac",   default=0.15, type=float)
    p.add_argument("--synthetic-n", default=_SYNTHETIC_N, type=int,
                   help="Synthetic samples if no CSV found")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    X, y = load_or_generate(args.data, n_synthetic=args.synthetic_n)
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_frac=args.val_frac)
    train(
        X_train, y_train, X_val, y_val,
        n_hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        out_path=args.out,
    )
