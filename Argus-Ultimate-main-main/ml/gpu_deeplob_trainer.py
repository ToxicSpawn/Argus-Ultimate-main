"""
gpu_deeplob_trainer.py — GPU-accelerated DeepLOB training pipeline for RTX 5080.

Trains the Conv1D → Inception-style module → LSTM → FC architecture described in
Zhang et al. (2019) "DeepLOB: Deep Learning for Limit Order Books"
https://arxiv.org/abs/1901.04716

Optimised for NVIDIA RTX 5080 16 GB GDDR7:
  • fp16 mixed precision via torch.cuda.amp (2× Tensor Core throughput)
  • batch_size=256 — fits comfortably in 16 GB VRAM
  • pin_memory=True DataLoaders for faster PCIe transfer
  • num_workers=8 leveraging the host CPU's 24 cores

Falls back transparently to CPU when CUDA is unavailable.
"""
from __future__ import annotations

import itertools
import logging
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional PyTorch ──────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, random_split

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    # Provide minimal stubs so module-level definitions don't crash at import time.
    import contextlib

    class Dataset:  # type: ignore
        pass

    class _FakeNoGrad:  # type: ignore
        """Decorator/context manager stub for torch.no_grad when torch is absent.

        Supports both usages:
          with torch.no_grad():  …   # context manager — __call__ returns self
          @torch.no_grad()           # decorator — __call__ returns a decorator
        When used as ``with torch.no_grad(): …`` the expression evaluates to an
        instance, so __enter__/__exit__ are invoked.  When used as a decorator
        ``@torch.no_grad()`` the result of __call__() must itself be callable
        (a decorator); we return a passthrough decorator in that case.
        """
        def __call__(self, fn=None):
            if callable(fn):
                # Used as @torch.no_grad()(fn) — passthrough
                return fn
            # Used as context manager: return self so __enter__/__exit__ fire
            return self
        def __enter__(self): return self
        def __exit__(self, *a): pass

    class _FakeCudaAmp:  # type: ignore
        @staticmethod
        def autocast(enabled=True):
            return contextlib.nullcontext()
        class GradScaler:  # type: ignore
            def __init__(self, enabled=True): pass
            def scale(self, loss): return loss
            def unscale_(self, opt): pass
            def step(self, opt): pass
            def update(self): pass

    class _FakeCuda:  # type: ignore
        amp = _FakeCudaAmp()
        @staticmethod
        def is_available(): return False
        @staticmethod
        def get_device_name(i=0): return ""
        @staticmethod
        def memory_reserved(i=0): return 0
        @staticmethod
        def memory_allocated(i=0): return 0
        @staticmethod
        def get_device_properties(i=0):
            return type("P", (), {"name": "none", "total_memory": 0})()
        @staticmethod
        def synchronize(): pass

    class _FakeVersion:  # type: ignore
        cuda = None

    class _FakeCudnn:  # type: ignore
        @staticmethod
        def version(): return 0

    class _FakeBackends:  # type: ignore
        cudnn = _FakeCudnn()

    class _FakeGenerator:  # type: ignore
        def manual_seed(self, s): return self

    class torch:  # type: ignore
        cuda = _FakeCuda()
        no_grad = _FakeNoGrad()
        version = _FakeVersion()
        backends = _FakeBackends()
        Generator = _FakeGenerator
        @staticmethod
        def device(s): return s
        @staticmethod
        def tensor(*a, **kw): return None
        @staticmethod
        def zeros(*a, **kw): return None
        @staticmethod
        def save(*a, **kw): pass
        @staticmethod
        def load(*a, **kw): return {}

    class nn:  # type: ignore
        class Module:
            def parameters(self): return iter([])
            def state_dict(self): return {}
            def load_state_dict(self, d): pass
            def to(self, dev): return self
            def eval(self): return self
            def train(self, mode=True): return self
        class CrossEntropyLoss:
            def __call__(self, *a, **kw): return type("L", (), {"item": lambda s: 0.0, "backward": lambda s: None})()
        class Linear:
            def __init__(self, *a, **kw): pass
        class LSTM:
            def __init__(self, *a, **kw): pass
        class Conv1d:
            def __init__(self, *a, **kw): pass
        class MaxPool1d:
            def __init__(self, *a, **kw): pass
        class BatchNorm1d:
            def __init__(self, *a, **kw): pass
        class LeakyReLU:
            def __init__(self, *a, **kw): pass
        class Dropout:
            def __init__(self, *a, **kw): pass
        class utils:
            @staticmethod
            def clip_grad_norm_(*a, **kw): pass

    logger.warning("gpu_deeplob_trainer: torch not available — training disabled")

# Optional pandas / pyarrow for data loading
try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ─── TrainerConfig ────────────────────────────────────────────────────────────


@dataclass
class TrainerConfig:
    """Configuration for the GPU-accelerated DeepLOB trainer."""

    device: str = "auto"                      # "auto" → CUDA if available else CPU
    model_path: str = "models/deeplob_weights.pt"
    batch_size: int = 256                     # RTX 5080 can handle large batches
    learning_rate: float = 0.001
    epochs: int = 50
    val_split: float = 0.2
    mixed_precision: bool = True              # fp16 on RTX 5080 Tensor Cores = 2× throughput
    num_workers: int = 8                      # DataLoader workers — host has 24 cores
    sequence_length: int = 100               # 100 LOB snapshots per sample
    feature_dim: int = 40                    # 10 levels × 4: bid_p, bid_s, ask_p, ask_s
    n_classes: int = 3                       # up, down, neutral
    early_stopping_patience: int = 5
    checkpoint_dir: str = "models/checkpoints"
    label_horizon: int = 10                  # predict direction at +10 ticks
    up_threshold: float = 0.001             # >0.1% mid move → UP
    down_threshold: float = -0.001          # <-0.1% mid move → DOWN
    gradient_clip: float = 1.0              # max gradient norm
    weight_decay: float = 1e-5


# ─── LOB Dataset ──────────────────────────────────────────────────────────────


class LOBSequenceDataset(Dataset):
    """Sliding-window LOB dataset.

    Produces (features, label) pairs where:
      features : float32 tensor of shape (sequence_length, feature_dim)
      label    : int64 tensor scalar in {0=up, 1=neutral, 2=down}
    """

    def __init__(
        self,
        features: np.ndarray,   # shape (T, feature_dim)
        labels: np.ndarray,     # shape (T,) int
        sequence_length: int,
    ) -> None:
        assert features.shape[0] == labels.shape[0], "features/labels length mismatch"
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.seq_len = sequence_length
        # valid starting indices: need seq_len past samples and label available
        self.n_valid = len(labels) - sequence_length

    def __len__(self) -> int:
        return max(0, self.n_valid)

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        window = self.features[idx : idx + self.seq_len]   # (seq_len, feat_dim)
        label = self.labels[idx + self.seq_len - 1]        # label at last step
        x = torch.from_numpy(window)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# ─── GPU-optimised DeepLOB model ─────────────────────────────────────────────


if _TORCH_AVAILABLE:

    class _InceptionModule(nn.Module):
        """Parallel multi-scale convolutions inspired by InceptionNet.

        Accepts input of shape (B, C_in, L) and concatenates outputs from
        three parallel conv branches.
        """

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            mid = out_channels // 4
            self.branch1 = nn.Sequential(
                nn.Conv1d(in_channels, mid, kernel_size=1, padding=0),
                nn.BatchNorm1d(mid),
                nn.LeakyReLU(0.01),
            )
            self.branch3 = nn.Sequential(
                nn.Conv1d(in_channels, mid, kernel_size=1),
                nn.BatchNorm1d(mid),
                nn.LeakyReLU(0.01),
                nn.Conv1d(mid, mid, kernel_size=3, padding=1),
                nn.BatchNorm1d(mid),
                nn.LeakyReLU(0.01),
            )
            self.branch5 = nn.Sequential(
                nn.Conv1d(in_channels, mid, kernel_size=1),
                nn.BatchNorm1d(mid),
                nn.LeakyReLU(0.01),
                nn.Conv1d(mid, mid, kernel_size=5, padding=2),
                nn.BatchNorm1d(mid),
                nn.LeakyReLU(0.01),
            )
            self.pool_branch = nn.Sequential(
                nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
                nn.Conv1d(in_channels, mid, kernel_size=1),
                nn.BatchNorm1d(mid),
                nn.LeakyReLU(0.01),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return torch.cat(
                [self.branch1(x), self.branch3(x), self.branch5(x), self.pool_branch(x)],
                dim=1,
            )

    class _GPUDeepLOBNet(nn.Module):
        """Full GPU-optimised DeepLOB architecture.

        Input  : (B, seq_len, feature_dim) — batch of LOB sliding windows
        Output : (B, n_classes)            — class logits

        Architecture
        ------------
        1. Conv1D feature extraction block (matches ml/deep_lob.py exactly)
           Conv1D(1→32, k=2) → LeakyReLU → Conv1D(32→32, k=2) → MaxPool(2)
           applied independently to each feature dimension via reshape trick
        2. Inception module (B, 32, L) → (B, 128, L) for richer multi-scale features
        3. Bi-LSTM(128→64, 2 layers) for temporal modelling
        4. Dropout(0.2)
        5. FC(128→n_classes)  — ×2 because bidirectional
        """

        def __init__(
            self,
            feature_dim: int = 40,
            n_classes: int = 3,
            dropout: float = 0.2,
        ) -> None:
            super().__init__()
            self.feature_dim = feature_dim

            # ─ Conv1D block (matches original ml/deep_lob.py) ─
            self.conv1 = nn.Conv1d(1, 32, kernel_size=2)
            self.conv2 = nn.Conv1d(32, 32, kernel_size=2)
            self.pool = nn.MaxPool1d(kernel_size=2)
            self.bn1 = nn.BatchNorm1d(32)
            self.bn2 = nn.BatchNorm1d(32)
            self.leaky = nn.LeakyReLU(0.01)

            # Compute conv output length for feature_dim input
            with torch.no_grad():
                probe = torch.zeros(1, 1, feature_dim)
                probe = self.pool(self.leaky(self.conv2(self.leaky(self.conv1(probe)))))
                conv_out_len = probe.shape[2]  # spatial dim after conv+pool

            # ─ Inception module ─
            self.inception = _InceptionModule(32, 128)

            # ─ Temporal: Bi-LSTM ─
            # After inception: (B, 128, conv_out_len) per feature snapshot
            # We feed (B, seq_len, 128*conv_out_len) as LSTM sequence
            self.lstm_input_size = 128 * conv_out_len
            self.lstm = nn.LSTM(
                input_size=self.lstm_input_size,
                hidden_size=64,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=dropout,
            )
            self.dropout = nn.Dropout(dropout)

            # ─ Classification head ─
            self.fc1 = nn.Linear(128, 64)  # 64 * 2 (bidirectional)
            self.fc2 = nn.Linear(64, n_classes)
            self.bn_fc = nn.BatchNorm1d(64)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Parameters
            ----------
            x : Tensor shape (B, seq_len, feature_dim)

            Returns
            -------
            logits : Tensor shape (B, n_classes)
            """
            B, T, D = x.shape
            # Reshape to apply conv over feature dimension per time step
            # → (B*T, 1, D)
            x = x.view(B * T, 1, D)

            # Conv1D feature extraction
            x = self.leaky(self.bn1(self.conv1(x)))    # (B*T, 32, D-1)
            x = self.leaky(self.bn2(self.conv2(x)))    # (B*T, 32, D-3)
            x = self.pool(x)                           # (B*T, 32, (D-3)//2)

            # Inception
            x = self.inception(x)                      # (B*T, 128, L)

            # Flatten spatial dims
            x = x.view(B, T, -1)                       # (B, T, 128*L)

            # LSTM over time
            x, _ = self.lstm(x)                        # (B, T, 128)
            x = x[:, -1, :]                            # last step: (B, 128)
            x = self.dropout(x)

            # Classification head
            x = F.relu(self.bn_fc(self.fc1(x)))        # (B, 64)
            return self.fc2(x)                         # (B, n_classes)


# ─── GPUDeepLOBTrainer ────────────────────────────────────────────────────────


class GPUDeepLOBTrainer:
    """GPU-accelerated DeepLOB training pipeline optimised for RTX 5080.

    Example
    -------
        config = TrainerConfig(epochs=50, mixed_precision=True)
        trainer = GPUDeepLOBTrainer(config)
        train_loader, val_loader = trainer.prepare_dataset("data/lob_train.parquet")
        trainer.build_model()
        result = trainer.train(train_loader, val_loader)
        trainer.save_model()
    """

    def __init__(self, config: Optional[TrainerConfig] = None) -> None:
        self.config = config or TrainerConfig()
        self.model: Optional["nn.Module"] = None
        self._scaler: Optional["torch.cuda.amp.GradScaler"] = None
        self._optimizer: Optional["torch.optim.Optimizer"] = None
        self._scheduler: Optional[object] = None

        # Resolve device
        self.device = self._resolve_device(self.config.device)
        logger.info("GPUDeepLOBTrainer: using device=%s", self.device)

        # Ensure checkpoint directory exists
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.model_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Device resolution ─────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device_str: str) -> str:
        """Resolve "auto" → "cuda:0" or "cpu"; validate explicit device strings."""
        if not _TORCH_AVAILABLE:
            if device_str not in ("auto", "cpu"):
                warnings.warn(
                    "gpu_deeplob_trainer: torch not available — falling back to cpu",
                    RuntimeWarning,
                )
            return "cpu"

        if device_str == "auto":
            if torch.cuda.is_available():
                dev = "cuda:0"
                logger.info(
                    "GPUDeepLOBTrainer: CUDA detected — %s",
                    torch.cuda.get_device_name(0),
                )
                return dev
            else:
                warnings.warn(
                    "gpu_deeplob_trainer: CUDA not available — training on CPU (slow)",
                    RuntimeWarning,
                )
                return "cpu"

        if device_str.startswith("cuda") and not torch.cuda.is_available():
            warnings.warn(
                f"gpu_deeplob_trainer: device '{device_str}' requested but CUDA unavailable "
                "— falling back to cpu",
                RuntimeWarning,
            )
            return "cpu"

        return device_str

    # ── Dataset preparation ───────────────────────────────────────────────────

    def _load_raw_data(self, data_path: str) -> np.ndarray:
        """Load LOB data from CSV or Parquet.

        Expected columns (in order, may also be named):
            bid_p_0..9, ask_p_0..9, bid_s_0..9, ask_s_0..9  (40 columns)
        OR a single 'features' column containing serialised 40-d vectors.

        Returns
        -------
        np.ndarray of shape (T, feature_dim)
        """
        if not _PANDAS_AVAILABLE:
            raise RuntimeError(
                "gpu_deeplob_trainer: pandas is required for data loading. "
                "Install with: pip install pandas"
            )

        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        suffix = path.suffix.lower()
        if suffix in (".parquet", ".pq"):
            df = pd.read_parquet(data_path)
        elif suffix in (".csv", ".txt"):
            df = pd.read_csv(data_path)
        elif suffix in (".feather",):
            df = pd.read_feather(data_path)
        else:
            # Try parquet first, fall back to CSV
            try:
                df = pd.read_parquet(data_path)
            except Exception:
                df = pd.read_csv(data_path)

        # Accept either 40 numeric columns or a pre-built feature matrix
        feature_cols = [c for c in df.columns if c not in ("label", "timestamp", "symbol")]
        if len(feature_cols) < self.config.feature_dim:
            raise ValueError(
                f"Expected at least {self.config.feature_dim} feature columns, "
                f"found {len(feature_cols)} in {data_path}"
            )
        raw = df[feature_cols[: self.config.feature_dim]].values.astype(np.float32)
        return raw

    def _normalise_features(self, raw: np.ndarray) -> np.ndarray:
        """Per-level price-relative normalisation.

        Prices are stored as:  [bid_p_0..9 | ask_p_0..9 | bid_s_0..9 | ask_s_0..9]
        This mirrors ml/deep_lob.py: prices normalised by mid, sizes by total depth.
        Applied row-wise (per snapshot).
        """
        n_levels = self.config.feature_dim // 4
        feat = raw.copy()

        # Extract columns by level convention
        bp = feat[:, :n_levels]           # bid prices
        ap = feat[:, n_levels : 2 * n_levels]  # ask prices
        bs = feat[:, 2 * n_levels : 3 * n_levels]  # bid sizes
        as_ = feat[:, 3 * n_levels :]    # ask sizes

        # Mid price from best bid/ask
        mid = (bp[:, 0] + ap[:, 0]) / 2.0 + 1e-12  # (T,)

        # Normalise prices relative to mid
        bp = (bp - mid[:, None]) / mid[:, None]
        ap = (ap - mid[:, None]) / mid[:, None]

        # Normalise sizes by total depth
        total_depth = bs.sum(axis=1) + as_.sum(axis=1) + 1e-12  # (T,)
        bs = bs / total_depth[:, None]
        as_ = as_ / total_depth[:, None]

        return np.concatenate([bp, ap, bs, as_], axis=1).astype(np.float32)

    def _compute_labels(self, raw: np.ndarray) -> np.ndarray:
        """Compute mid-price direction labels at +horizon ticks.

        Returns
        -------
        np.ndarray of shape (T,) with values:
            0 = up   (future mid > current mid × (1 + up_threshold))
            1 = neutral
            2 = down (future mid < current mid × (1 + down_threshold))
        """
        n_levels = self.config.feature_dim // 4
        H = self.config.label_horizon

        # Best bid and ask are the first columns
        best_bid = raw[:, 0]
        best_ask = raw[:, n_levels]
        mid = (best_bid + best_ask) / 2.0

        labels = np.ones(len(mid), dtype=np.int64)  # default: neutral
        for i in range(len(mid) - H):
            ret = (mid[i + H] - mid[i]) / (mid[i] + 1e-12)
            if ret > self.config.up_threshold:
                labels[i] = 0  # up
            elif ret < self.config.down_threshold:
                labels[i] = 2  # down
            # else neutral (1)

        # Last H entries have no valid label — mark neutral
        labels[-H:] = 1
        return labels

    def prepare_dataset(
        self, data_path: str
    ) -> Tuple["DataLoader", "DataLoader"]:
        """Load, normalise, and split a LOB dataset into train/val DataLoaders.

        Parameters
        ----------
        data_path : str
            Path to a CSV or Parquet file of raw LOB snapshots.

        Returns
        -------
        (train_loader, val_loader) — DataLoader with pin_memory=True for fast GPU transfer
        """
        logger.info("GPUDeepLOBTrainer: loading data from %s", data_path)
        raw = self._load_raw_data(data_path)
        features = self._normalise_features(raw)
        labels = self._compute_labels(raw)

        dataset = LOBSequenceDataset(features, labels, self.config.sequence_length)
        n_total = len(dataset)
        n_val = max(1, int(n_total * self.config.val_split))
        n_train = n_total - n_val

        logger.info(
            "GPUDeepLOBTrainer: dataset size=%d  train=%d  val=%d",
            n_total, n_train, n_val,
        )

        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required for DataLoader creation")

        train_ds, val_ds = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        use_pin = self.device.startswith("cuda")
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=use_pin,
            drop_last=True,
            persistent_workers=self.config.num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=use_pin,
            drop_last=False,
            persistent_workers=self.config.num_workers > 0,
        )
        return train_loader, val_loader

    # ── Model construction ────────────────────────────────────────────────────

    def build_model(self) -> "nn.Module":
        """Construct and move the GPU-optimised DeepLOB model to device.

        Architecture mirrors ml/deep_lob.py (Conv1D → LSTM → FC) but adds:
          • BatchNorm after each conv
          • Inception multi-scale module between conv and LSTM
          • Bi-LSTM (2 layers) for richer temporal modelling
          • Dropout(0.2) before the classification head

        Returns
        -------
        nn.Module on self.device
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required to build the model")

        self.model = _GPUDeepLOBNet(
            feature_dim=self.config.feature_dim,
            n_classes=self.config.n_classes,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            "GPUDeepLOBTrainer: model built — %d trainable parameters on %s",
            n_params,
            self.device,
        )
        return self.model

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        train_loader: "DataLoader",
        val_loader: "DataLoader",
    ) -> Dict:
        """Full training loop with mixed precision, early stopping, and checkpointing.

        Parameters
        ----------
        train_loader, val_loader : DataLoader

        Returns
        -------
        dict with keys:
            epochs_trained, best_val_loss, best_val_accuracy, training_time_s
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required for training")
        if self.model is None:
            self.build_model()

        use_amp = self.config.mixed_precision and self.device.startswith("cuda")
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self._optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            patience=2,
            factor=0.5,
            min_lr=1e-6,
        )
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_val_acc = 0.0
        patience_counter = 0
        best_ckpt_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
        t0 = time.time()

        for epoch in range(1, self.config.epochs + 1):
            # ── Train ─
            train_loss, train_acc = self._train_epoch(
                train_loader, criterion, use_amp
            )

            # ── Validate ─
            val_loss, val_acc = self._val_epoch(val_loader, criterion, use_amp)

            # ── LR scheduler ─
            self._scheduler.step(val_loss)
            current_lr = self._optimizer.param_groups[0]["lr"]

            # ── GPU memory ─
            gpu_info = self.get_gpu_stats()
            vram_used = gpu_info.get("vram_used_mb", 0)

            logger.info(
                "Epoch %d/%d | train_loss=%.4f acc=%.3f | val_loss=%.4f acc=%.3f "
                "| lr=%.2e | vram=%dMB",
                epoch, self.config.epochs,
                train_loss, train_acc,
                val_loss, val_acc,
                current_lr,
                vram_used,
            )

            # ── Checkpoint ─
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0
                self._save_checkpoint(best_ckpt_path, epoch, best_val_loss)
                logger.info(
                    "GPUDeepLOBTrainer: checkpoint saved (val_loss=%.4f)", best_val_loss
                )
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(
                        "GPUDeepLOBTrainer: early stopping after %d epochs (patience=%d)",
                        epoch,
                        self.config.early_stopping_patience,
                    )
                    break

        # Load best checkpoint
        if os.path.exists(best_ckpt_path):
            self._load_checkpoint(best_ckpt_path)

        training_time = time.time() - t0
        logger.info(
            "GPUDeepLOBTrainer: training complete in %.1fs | best_val_loss=%.4f | "
            "best_val_acc=%.3f",
            training_time, best_val_loss, best_val_acc,
        )
        return {
            "epochs_trained": epoch,
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_acc,
            "training_time_s": training_time,
        }

    def _train_epoch(
        self,
        loader: "DataLoader",
        criterion: "nn.Module",
        use_amp: bool,
    ) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self._optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = self.model(x)
                loss = criterion(logits, y)

            self._scaler.scale(loss).backward()
            self._scaler.unscale_(self._optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )
            self._scaler.step(self._optimizer)
            self._scaler.update()

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += batch_size

        return total_loss / max(total, 1), correct / max(total, 1)

    def _val_epoch(
        self,
        loader: "DataLoader",
        criterion: "nn.Module",
        use_amp: bool,
    ) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = self.model(x)
                    loss = criterion(logits, y)

                batch_size = y.size(0)
                total_loss += loss.item() * batch_size
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += batch_size

        return total_loss / max(total, 1), correct / max(total, 1)

    # ── Evaluation ────────────────────────────────────────────────────────────

    def evaluate(self, val_loader: "DataLoader") -> Dict:
        """Compute per-class precision, recall, F1, and confusion matrix.

        Returns
        -------
        dict with keys: accuracy, per_class (dict), confusion_matrix
        """
        if not _TORCH_AVAILABLE or self.model is None:
            return {"error": "model not loaded"}

        self.model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []
        use_amp = self.config.mixed_precision and self.device.startswith("cuda")

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = self.model(x)
                preds = logits.argmax(dim=1).cpu().numpy().tolist()
                all_preds.extend(preds)
                all_labels.extend(y.numpy().tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        accuracy = float((y_true == y_pred).mean())

        result: Dict = {"accuracy": accuracy}

        if _SKLEARN_AVAILABLE:
            labels_list = [0, 1, 2]
            class_names = ["up", "neutral", "down"]
            result["per_class"] = {}
            prec = precision_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0)
            rec = recall_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0)
            f1 = f1_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0)
            for i, name in enumerate(class_names):
                result["per_class"][name] = {
                    "precision": float(prec[i]),
                    "recall": float(rec[i]),
                    "f1": float(f1[i]),
                }
            cm = confusion_matrix(y_true, y_pred, labels=labels_list)
            result["confusion_matrix"] = cm.tolist()
            result["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        else:
            # Basic per-class accuracy without sklearn
            for i, name in enumerate(["up", "neutral", "down"]):
                mask = y_true == i
                if mask.sum() > 0:
                    cls_acc = float((y_pred[mask] == i).mean())
                else:
                    cls_acc = 0.0
                result[f"class_{name}_accuracy"] = cls_acc

        return result

    # ── Checkpointing ─────────────────────────────────────────────────────────

    def _save_checkpoint(self, path: str, epoch: int, val_loss: float) -> None:
        if not _TORCH_AVAILABLE or self.model is None:
            return
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": {
                    "feature_dim": self.config.feature_dim,
                    "n_classes": self.config.n_classes,
                    "sequence_length": self.config.sequence_length,
                },
                "model_version": "v1",
            },
            path,
        )

    def _load_checkpoint(self, path: str) -> None:
        if not _TORCH_AVAILABLE or self.model is None:
            return
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        logger.info(
            "GPUDeepLOBTrainer: loaded checkpoint from %s (epoch=%d val_loss=%.4f)",
            path, ckpt.get("epoch", -1), ckpt.get("val_loss", float("inf")),
        )

    def save_model(self, path: Optional[str] = None) -> None:
        """Save model weights to path (defaults to config.model_path)."""
        if not _TORCH_AVAILABLE or self.model is None:
            logger.warning("GPUDeepLOBTrainer: no model to save")
            return
        save_path = path or self.config.model_path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": {
                    "feature_dim": self.config.feature_dim,
                    "n_classes": self.config.n_classes,
                    "sequence_length": self.config.sequence_length,
                },
                "trained": True,
                "model_version": "v1",
            },
            save_path,
        )
        logger.info("GPUDeepLOBTrainer: model saved to %s", save_path)

    # ── GPU diagnostics ───────────────────────────────────────────────────────

    def get_gpu_stats(self) -> Dict:
        """Return GPU memory, utilisation, and CUDA version information.

        Returns
        -------
        dict with keys:
            available, gpu_name, vram_total_mb, vram_used_mb, vram_free_mb,
            cuda_version, cudnn_version, utilisation_pct
        """
        if not _TORCH_AVAILABLE or not torch.cuda.is_available():
            return {"available": False}

        try:
            device_idx = 0
            if self.device.startswith("cuda:"):
                device_idx = int(self.device.split(":")[1])

            props = torch.cuda.get_device_properties(device_idx)
            total_mb = props.total_memory // (1024 * 1024)
            reserved_mb = torch.cuda.memory_reserved(device_idx) // (1024 * 1024)
            allocated_mb = torch.cuda.memory_allocated(device_idx) // (1024 * 1024)
            free_mb = total_mb - reserved_mb

            # NVML utilisation (optional, may not be available)
            util_pct: Optional[float] = None
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                util_pct = float(util.gpu)
            except Exception:
                pass

            return {
                "available": True,
                "gpu_name": props.name,
                "vram_total_mb": total_mb,
                "vram_used_mb": allocated_mb,
                "vram_reserved_mb": reserved_mb,
                "vram_free_mb": free_mb,
                "cuda_version": torch.version.cuda or "unknown",
                "cudnn_version": str(torch.backends.cudnn.version()),
                "utilisation_pct": util_pct,
                "device_index": device_idx,
            }
        except Exception as exc:
            logger.debug("GPUDeepLOBTrainer.get_gpu_stats error: %s", exc)
            return {"available": True, "error": str(exc)}
