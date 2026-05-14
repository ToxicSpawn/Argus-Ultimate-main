"""
TFT Training Pipeline — Walk-forward retraining for the Temporal Fusion Transformer.

Fetches historical OHLCV from Kraken (via ccxt), trains _NumpyTFT on in-sample
data, validates on OOS, saves weights to ml/weights/, and logs Sharpe + regime accuracy.

Usage:
    python -m ml.training.train_tft --symbol BTC/USD --timeframe 1h --months 18
    python -m ml.training.train_tft --symbol BTC/USD --timeframe 1h --months 18 --epochs 50

Scheduled retraining: call retrain_all_symbols() from a cron job or APScheduler.
"""
from __future__ import annotations

import json
import logging
import math
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy dependencies — all wrapped in try/except
# ---------------------------------------------------------------------------

try:
    from ml.models.temporal_fusion_transformer import _NumpyTFT, TemporalFusionTransformer
except ImportError:
    _NumpyTFT = None
    TemporalFusionTransformer = None

try:
    import ccxt
    _CCXT_AVAILABLE = True
except ImportError:
    ccxt = None  # type: ignore[assignment]
    _CCXT_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression, Ridge
    _SKLEARN_AVAILABLE = True
except ImportError:
    LogisticRegression = None  # type: ignore[assignment]
    Ridge = None  # type: ignore[assignment]
    _SKLEARN_AVAILABLE = False

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    pd = None  # type: ignore[assignment]
    _PANDAS_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
    _CUDA_AVAILABLE = torch.cuda.is_available()
    if _CUDA_AVAILABLE:
        logger.info(
            "TFT trainer: CUDA available — %s (%dMB)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory // 2**20,
        )
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    _CUDA_AVAILABLE = False

# Default compute device for PyTorch operations
TORCH_DEVICE = "cuda" if _CUDA_AVAILABLE else "cpu"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEIGHTS_DIR = Path("ml/weights")
LABEL_LOOKBACK = 24          # hours — labels generated from future realized vol + trend
HORIZONS = [6, 12, 24, 48]  # label horizons
MIN_BARS = 500               # minimum bars to start training
DEFAULT_EPOCHS = 30
DEFAULT_LR = 0.001
DEFAULT_HIDDEN = 64
REGIME_LABELS = ["TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOL", "CRISIS"]

# Feature count must match _NumpyTFT expectation (13 features)
N_FEATURES = 13
SEQ_LEN = 60  # sliding window length for ELM body

# ---------------------------------------------------------------------------
# OHLCV fetching
# ---------------------------------------------------------------------------


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1h",
    months: int = 18,
    exchange_id: str = "kraken",
) -> Optional[Any]:
    """
    Fetch historical OHLCV from Kraken (or another ccxt exchange).

    Returns a pandas DataFrame with columns:
        timestamp, open, high, low, close, volume

    Falls back to synthetic sine-wave OHLCV if ccxt is unavailable or on error.
    Returns None only on unrecoverable internal error.
    """
    if not _PANDAS_AVAILABLE:
        logger.error("pandas is required for fetch_ohlcv; returning None")
        return None

    # Determine how far back to go
    now_ms = int(time.time() * 1000)
    lookback_ms = months * 30 * 24 * 3600 * 1000
    since_ms = now_ms - lookback_ms

    # Map timeframe string to milliseconds per bar
    _tf_to_ms = {
        "1m": 60_000, "5m": 300_000, "15m": 900_000,
        "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000,
        "1d": 86_400_000,
    }
    bar_ms = _tf_to_ms.get(timeframe, 3_600_000)
    expected_bars = lookback_ms // bar_ms

    if _CCXT_AVAILABLE:
        try:
            exchange_cls = getattr(ccxt, exchange_id, None)
            if exchange_cls is None:
                logger.warning("ccxt exchange '%s' not found; using synthetic data", exchange_id)
            else:
                exchange = exchange_cls({"enableRateLimit": True})
                all_ohlcv: List[List] = []
                cursor = since_ms

                # Paginate — ccxt returns at most 500–1000 bars per call
                max_iters = int(math.ceil(expected_bars / 500)) + 5
                for _ in range(max_iters):
                    try:
                        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=500)
                    except Exception as fetch_err:
                        logger.warning("ccxt fetch_ohlcv error: %s", fetch_err)
                        break

                    if not batch:
                        break

                    all_ohlcv.extend(batch)
                    last_ts = batch[-1][0]

                    if last_ts >= now_ms - bar_ms:
                        break  # reached the present

                    if last_ts <= cursor:
                        break  # no progress

                    cursor = last_ts + bar_ms

                if all_ohlcv:
                    df = pd.DataFrame(
                        all_ohlcv,
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
                    df[["open", "high", "low", "close", "volume"]] = df[
                        ["open", "high", "low", "close", "volume"]
                    ].astype(float)
                    logger.info(
                        "Fetched %d bars for %s/%s from %s",
                        len(df), symbol, timeframe, exchange_id,
                    )
                    return df

                logger.warning("No OHLCV data returned from ccxt for %s; falling back to synthetic", symbol)

        except Exception as exc:
            logger.warning("ccxt initialisation error (%s); falling back to synthetic: %s", exchange_id, exc)
    else:
        logger.info("ccxt not available; generating synthetic OHLCV for testing")

    # --- Synthetic fallback ---
    return _make_synthetic_ohlcv(symbol, timeframe, months)


def _make_synthetic_ohlcv(symbol: str, timeframe: str, months: int) -> Any:
    """Generate a synthetic sine-wave OHLCV DataFrame for unit-testing."""
    _tf_to_hours = {
        "1m": 1 / 60, "5m": 5 / 60, "15m": 0.25,
        "30m": 0.5, "1h": 1.0, "4h": 4.0, "1d": 24.0,
    }
    hours_per_bar = _tf_to_hours.get(timeframe, 1.0)
    n_bars = int((months * 30 * 24) / hours_per_bar)
    n_bars = max(n_bars, MIN_BARS + 100)

    rng = np.random.default_rng(seed=abs(hash(symbol)) % (2**31))
    t = np.arange(n_bars)

    # Trend + cycle + noise
    price = 30_000 + 5_000 * np.sin(2 * math.pi * t / (24 * 30))  # monthly cycle
    price += 2_000 * np.sin(2 * math.pi * t / (24 * 7))            # weekly cycle
    price += rng.normal(0, 200, n_bars).cumsum() * 0.1              # random walk
    price = np.maximum(price, 100.0)

    noise_pct = rng.uniform(0.001, 0.008, n_bars)
    high = price * (1 + noise_pct)
    low = price * (1 - noise_pct)
    open_ = np.roll(price, 1)
    open_[0] = price[0]
    volume = rng.lognormal(mean=10, sigma=1.5, size=n_bars)

    now = int(time.time())
    bar_seconds = int(hours_per_bar * 3600)
    timestamps = pd.to_datetime(
        [now - (n_bars - i) * bar_seconds for i in range(n_bars)],
        unit="s", utc=True,
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": price,
        "volume": volume,
    })
    logger.info("Generated %d synthetic bars for %s/%s", len(df), symbol, timeframe)
    return df


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------


def generate_labels(df: Any) -> np.ndarray:
    """
    Generate integer regime labels for each bar in df.

    For each bar i:
      - future_ret_24h  = (close[i+24] / close[i]) - 1
      - realized_vol_24h = std(log_returns[i : i+24])
      - trend_strength   = |future_ret_24h| / (realized_vol_24h + 1e-8)

    Label mapping (priority order):
      vol > 0.03 AND future_ret < -0.02  -> CRISIS    (4)
      vol > 0.025                         -> HIGH_VOL  (3)
      trend_strength > 1.5 AND ret > 0   -> TREND_UP  (0)
      trend_strength > 1.5 AND ret < 0   -> TREND_DOWN(1)
      else                                -> RANGE     (2)

    The last 48 bars cannot be labelled (not enough future data) — set to RANGE.
    """
    closes = df["close"].values.astype(float)
    n = len(closes)

    log_returns = np.zeros(n)
    log_returns[1:] = np.log(closes[1:] / np.maximum(closes[:-1], 1e-10))

    labels = np.full(n, 2, dtype=np.int32)  # default RANGE

    horizon = LABEL_LOOKBACK  # 24
    for i in range(n - horizon - 24):  # leave 48-bar buffer at the end
        future_slice = closes[i + 1: i + horizon + 1]
        if len(future_slice) < horizon:
            break

        future_ret = (future_slice[-1] / max(closes[i], 1e-10)) - 1.0
        ret_window = log_returns[i: i + horizon]
        realized_vol = float(np.std(ret_window)) if len(ret_window) > 1 else 0.0
        trend_strength = abs(future_ret) / (realized_vol + 1e-8)

        if realized_vol > 0.03 and future_ret < -0.02:
            labels[i] = 4  # CRISIS
        elif realized_vol > 0.025:
            labels[i] = 3  # HIGH_VOL
        elif trend_strength > 1.5 and future_ret > 0:
            labels[i] = 0  # TREND_UP
        elif trend_strength > 1.5 and future_ret < 0:
            labels[i] = 1  # TREND_DOWN
        else:
            labels[i] = 2  # RANGE

    return labels


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------


def build_feature_matrix(df: Any) -> np.ndarray:
    """
    Build a (N, 13) float32 feature matrix from OHLCV DataFrame.

    Features (index 0–12):
      0  log return 1 bar
      1  log volume
      2  RSI(14)
      3  MACD / close
      4  Bollinger %B
      5  ATR(14) / close
      6  Momentum(10)
      7  Body ratio
      8  High-Low ratio
      9  Volume ratio (vol / 20-bar MA)
      10 Hour of day (0–1)
      11 Day of week (0–1)
      12 Funding rate placeholder (0.0)
    """
    close = df["close"].values.astype(float)
    open_ = df["open"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)
    n = len(close)

    # --- Feature 0: log return ---
    log_ret = np.zeros(n)
    log_ret[1:] = np.log(np.maximum(close[1:], 1e-10) / np.maximum(close[:-1], 1e-10))

    # --- Feature 1: log volume ---
    log_vol = np.log(volume + 1.0)

    # --- Feature 2: RSI(14) ---
    rsi = _compute_rsi(close, period=14)

    # --- Feature 3: MACD / close ---
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = (ema12 - ema26) / np.maximum(close, 1e-10)

    # --- Feature 4: Bollinger %B ---
    sma20 = _rolling_mean(close, 20)
    std20 = _rolling_std(close, 20)
    bb_pct_b = (close - sma20) / (2.0 * std20 + 1e-10)

    # --- Feature 5: ATR(14) / close ---
    atr = _compute_atr(high, low, close, period=14)
    atr_pct = atr / np.maximum(close, 1e-10)

    # --- Feature 6: Momentum(10) ---
    mom10 = np.zeros(n)
    mom10[10:] = (close[10:] - close[:-10]) / np.maximum(close[:-10], 1e-10)

    # --- Feature 7: Body ratio ---
    body_ratio = (close - open_) / (high - low + 1e-8)

    # --- Feature 8: High-Low ratio ---
    hl_ratio = (high - low) / np.maximum(close, 1e-10)

    # --- Feature 9: Volume ratio ---
    vol_ma20 = _rolling_mean(volume, 20)
    vol_ratio = volume / (vol_ma20 + 1e-10)

    # --- Features 10–11: Temporal (hour, day of week) ---
    timestamps = df["timestamp"]
    if hasattr(timestamps, "dt"):
        hour = timestamps.dt.hour.values.astype(float)
        dow = timestamps.dt.dayofweek.values.astype(float)
    else:
        hour = np.zeros(n)
        dow = np.zeros(n)
    hour_norm = hour / 23.0
    dow_norm = dow / 6.0

    # --- Feature 12: Funding rate placeholder ---
    funding = np.zeros(n)

    # Stack into (N, 13)
    X = np.column_stack([
        log_ret, log_vol, rsi, macd, bb_pct_b,
        atr_pct, mom10, body_ratio, hl_ratio, vol_ratio,
        hour_norm, dow_norm, funding,
    ]).astype(np.float32)

    # Forward-fill NaN, then fill remaining with 0
    for col_idx in range(X.shape[1]):
        col = X[:, col_idx]
        nan_mask = np.isnan(col) | np.isinf(col)
        if nan_mask.any():
            # Forward fill
            last_valid = 0.0
            for row_idx in range(len(col)):
                if not nan_mask[row_idx]:
                    last_valid = col[row_idx]
                else:
                    col[row_idx] = last_valid
            # Any remaining NaN (at start) → 0
            col[np.isnan(col) | np.isinf(col)] = 0.0
            X[:, col_idx] = col

    return X


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    result = np.empty_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    result = np.full_like(arr, np.nan)
    cumsum = np.cumsum(arr)
    result[window - 1:] = (cumsum[window - 1:] - np.concatenate([[0], cumsum[:-window]])) / window
    # Fill initial NaN with expanding mean
    for i in range(window - 1):
        result[i] = arr[: i + 1].mean()
    return result


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    result = np.full(len(arr), 1e-8)
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        result[i] = arr[start: i + 1].std() + 1e-8
    return result


def _compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = _ema(gain, period)
    avg_loss = _ema(loss, period)

    rs = avg_gain / np.maximum(avg_loss, 1e-10)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi / 100.0  # normalise to [0, 1]


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return _ema(tr, period)


# ---------------------------------------------------------------------------
# Walk-forward trainer
# ---------------------------------------------------------------------------


class WalkForwardTrainer:
    """
    Walk-forward TFT retraining.

    Splits data into (train_months) in-sample + (oos_months) out-of-sample,
    trains an Extreme Learning Machine on the _NumpyTFT body representations,
    evaluates on OOS, and saves weights.
    """

    def __init__(
        self,
        symbol: str,
        timeframe: str = "1h",
        train_months: int = 18,
        oos_months: int = 3,
        epochs: int = DEFAULT_EPOCHS,
        hidden_dim: int = DEFAULT_HIDDEN,
        lr: float = DEFAULT_LR,
        save_weights: bool = True,
        # Phase W4: masked return SSL pretraining
        ssl_pretrain: bool = False,
        ssl_steps: int = 200,
        ssl_mask_prob: float = 0.15,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.train_months = train_months
        self.oos_months = oos_months
        self.epochs = epochs
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.save_weights = save_weights
        self.ssl_pretrain = bool(ssl_pretrain)
        self.ssl_steps = int(ssl_steps)
        self.ssl_mask_prob = float(ssl_mask_prob)
        self._symbol_safe = symbol.replace("/", "_").replace(":", "_")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        """Full walk-forward training run. Returns a metrics dict."""
        total_months = self.train_months + self.oos_months
        logger.info(
            "Starting walk-forward training for %s/%s (%d months total)",
            self.symbol, self.timeframe, total_months,
        )

        df = fetch_ohlcv(self.symbol, self.timeframe, total_months)
        if df is None or len(df) < MIN_BARS:
            msg = f"Insufficient data for {self.symbol}: got {0 if df is None else len(df)} bars (need {MIN_BARS})"
            logger.error(msg)
            return {"error": msg, "symbol": self.symbol, "timeframe": self.timeframe}

        # Build feature matrix and labels
        X = build_feature_matrix(df)
        y = generate_labels(df)

        # Phase W4: optional masked-return SSL pretraining run.
        # Produces a separate pretrained reconstruction head that the
        # downstream trainer can use to warm-start (future work). Currently
        # only logs pretraining metrics — wiring the warm-started weights
        # into ``_NumpyTFT`` is left as a separate change to keep this
        # commit focused on SSL library availability.
        if self.ssl_pretrain:
            try:
                from ml.ssl import MaskedReturnDataset, MaskedReturnPretrainer
                # Extract raw returns from ohlcv dataframe
                try:
                    closes = df["close"].values
                except Exception:
                    closes = np.asarray(df.iloc[:, 3], dtype=float)  # fallback
                returns = np.diff(np.log(np.clip(closes, 1e-9, None)))

                ssl_seq_len = 64
                if len(returns) >= ssl_seq_len * 2:
                    ds = MaskedReturnDataset(
                        sequences=[returns],
                        seq_len=ssl_seq_len,
                        mask_prob=self.ssl_mask_prob,
                    )
                    pretrainer = MaskedReturnPretrainer(
                        seq_len=ssl_seq_len,
                        learning_rate=1e-3,
                    )
                    losses = pretrainer.fit(ds, n_steps=self.ssl_steps, batch_size=16)
                    logger.info(
                        "SSL pretraining %s: %d steps, loss %.6f -> %.6f",
                        self.symbol, self.ssl_steps, losses[0], losses[-1],
                    )
                else:
                    logger.info(
                        "SSL pretraining skipped for %s: %d returns (need %d)",
                        self.symbol, len(returns), ssl_seq_len * 2,
                    )
            except Exception as exc:
                logger.warning("SSL pretraining failed for %s: %s", self.symbol, exc)

        n = len(X)
        # Number of bars that correspond to in-sample vs OOS
        # We approximate: oos_fraction = oos_months / total_months
        oos_frac = self.oos_months / total_months
        split_idx = int(n * (1.0 - oos_frac))
        split_idx = max(split_idx, MIN_BARS)

        if split_idx >= n - 50:
            logger.warning(
                "Not enough OOS bars for %s; adjusting split", self.symbol
            )
            split_idx = max(n - 50, MIN_BARS)

        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]

        logger.info(
            "Train bars: %d | OOS bars: %d | Classes: %s",
            len(X_train), len(X_val),
            {REGIME_LABELS[i]: int((y_train == i).sum()) for i in range(5)},
        )

        model, metrics = self._train_fold(X_train, y_train, X_val, y_val)

        if self.save_weights and model is not None:
            self._save_weights(model, metrics)

        metrics["symbol"] = self.symbol
        metrics["timeframe"] = self.timeframe
        metrics["train_bars"] = int(len(X_train))
        metrics["oos_bars"] = int(len(X_val))
        return metrics

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _train_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[Optional[Any], Dict]:
        """
        Train one fold using the Extreme Learning Machine approach:

        1. Initialise _NumpyTFT with fixed random body weights (seed=42).
        2. Forward-pass all training sequences through the body to get
           pooled representations (shape: N_seq x hidden_dim).
        3. Fit a LogisticRegression on pooled reps → regime labels.
        4. Fit a Ridge regression on pooled reps → direction labels (+1 / -1).
        5. Evaluate on OOS sequences.
        """
        if _NumpyTFT is None:
            logger.error("_NumpyTFT not available (ml.models not importable)")
            return None, {"error": "ml.models unavailable"}

        if not _SKLEARN_AVAILABLE:
            logger.error("scikit-learn not available; cannot train ELM classifier")
            return None, {"error": "scikit-learn unavailable"}

        # -- Build TFT body --
        model = _NumpyTFT(input_size=N_FEATURES, hidden=self.hidden_dim)

        # -- Extract pooled representations from sliding windows --
        logger.info("Extracting body representations from %d training bars...", len(X_train))
        X_body_train, y_body_train = self._extract_pooled(model, X_train, y_train)
        X_body_val, y_body_val = self._extract_pooled(model, X_val, y_val)

        if len(X_body_train) < 10:
            return None, {"error": "Too few training sequences after windowing"}

        # -- Direction labels: +1 if TREND_UP, -1 if TREND_DOWN, 0 otherwise --
        dir_train = np.where(y_body_train == 0, 1.0,
                    np.where(y_body_train == 1, -1.0, 0.0))
        dir_val = np.where(y_body_val == 0, 1.0,
                  np.where(y_body_val == 1, -1.0, 0.0))

        # -- Fit regime classifier (LogisticRegression) --
        logger.info("Fitting LogisticRegression on %d pooled representations...", len(X_body_train))
        regime_clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial",
            class_weight="balanced",
            random_state=42,
        )
        regime_clf.fit(X_body_train, y_body_train)

        # -- Fit direction regressor (Ridge) --
        dir_clf = Ridge(alpha=1.0)
        dir_clf.fit(X_body_train, dir_train)

        # -- Store classifiers on the model object for serialisation --
        model._regime_clf = regime_clf
        model._dir_clf = dir_clf

        # -- Evaluate on OOS --
        train_metrics = self._evaluate_with_clf(regime_clf, dir_clf, X_body_train, y_body_train, dir_train)
        val_metrics = self._evaluate_with_clf(regime_clf, dir_clf, X_body_val, y_body_val, dir_val)

        logger.info(
            "Train accuracy: %.3f | OOS accuracy: %.3f | OOS Sharpe proxy: %.3f",
            train_metrics["accuracy"],
            val_metrics["accuracy"],
            val_metrics.get("sharpe_proxy", float("nan")),
        )

        metrics: Dict[str, Any] = {
            "train_accuracy": float(train_metrics["accuracy"]),
            "oos_accuracy": float(val_metrics["accuracy"]),
            "oos_per_class_accuracy": val_metrics.get("per_class_accuracy", {}),
            "oos_sharpe_proxy": float(val_metrics.get("sharpe_proxy", 0.0)),
        }
        return model, metrics

    def _extract_pooled(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Slide a window of length SEQ_LEN over X, forward-pass each window
        through the _NumpyTFT body, and collect pooled representations.

        Returns (X_pooled, y_pooled) where y_pooled[i] is the label at
        the last bar of window i.
        """
        reps = []
        labels = []
        n = len(X)

        for i in range(SEQ_LEN, n):
            window = X[i - SEQ_LEN: i]  # (SEQ_LEN, N_FEATURES)
            try:
                h = model._vsn(window)   # (SEQ_LEN, hidden)
                h = model._grn(h)
                h = model._attention(h)
                # Pool: last + mean
                pooled = (h[-1] + h.mean(axis=0)) / 2.0  # (hidden,)
                reps.append(pooled)
                labels.append(int(y[i]))
            except Exception as exc:
                logger.debug("Skipping window %d due to error: %s", i, exc)
                continue

        if not reps:
            return np.empty((0, self.hidden_dim)), np.empty(0, dtype=np.int32)

        return np.array(reps, dtype=np.float32), np.array(labels, dtype=np.int32)

    def _evaluate(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate _NumpyTFT using its built-in forward() on sliding windows.
        Returns accuracy + per-class accuracy + Sharpe proxy.
        """
        if model is None:
            return {}

        preds = []
        for i in range(SEQ_LEN, len(X)):
            window = X[i - SEQ_LEN: i]
            try:
                result = model.forward(window)
                preds.append(REGIME_LABELS.index(result["regime"]))
            except Exception:
                preds.append(2)  # RANGE fallback

        y_true = y[SEQ_LEN:]
        preds_arr = np.array(preds[:len(y_true)], dtype=np.int32)

        acc = float((preds_arr == y_true[:len(preds_arr)]).mean()) if len(preds_arr) > 0 else 0.0
        per_class: Dict[str, float] = {}
        for idx, name in enumerate(REGIME_LABELS):
            mask = y_true[:len(preds_arr)] == idx
            if mask.sum() > 0:
                per_class[name] = float((preds_arr[mask] == idx).mean())

        sharpe = _compute_sharpe_proxy(preds_arr, y_true[:len(preds_arr)])

        return {"accuracy": acc, "per_class_accuracy": per_class, "sharpe_proxy": sharpe}

    def _evaluate_with_clf(
        self,
        regime_clf: Any,
        dir_clf: Any,
        X_body: np.ndarray,
        y_true: np.ndarray,
        dir_true: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate the fitted sklearn classifiers on body representations."""
        if len(X_body) == 0:
            return {"accuracy": 0.0, "per_class_accuracy": {}, "sharpe_proxy": 0.0}

        preds = regime_clf.predict(X_body)
        acc = float((preds == y_true).mean())

        per_class: Dict[str, float] = {}
        for idx, name in enumerate(REGIME_LABELS):
            mask = y_true == idx
            if mask.sum() > 0:
                per_class[name] = float((preds[mask] == idx).mean())

        sharpe = _compute_sharpe_proxy(preds, y_true)

        return {"accuracy": acc, "per_class_accuracy": per_class, "sharpe_proxy": sharpe}

    def _save_weights(self, model: Any, metrics: Dict[str, Any]) -> None:
        """
        Save _NumpyTFT body weights + sklearn classifiers to
        ml/weights/{symbol_safe}_{timeframe}.npz + .pkl
        """
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        stem = f"{self._symbol_safe}_{self.timeframe}"
        npz_path = WEIGHTS_DIR / f"{stem}.npz"
        pkl_path = WEIGHTS_DIR / f"{stem}_clf.pkl"
        meta_path = WEIGHTS_DIR / f"{stem}_meta.json"

        # Save numpy body arrays
        body_arrays: Dict[str, np.ndarray] = {
            "vsn_w": model.vsn_w,
            "vsn_v": model.vsn_v,
            "grn_w1": model.grn_w1,
            "grn_w2": model.grn_w2,
            "grn_gate": model.grn_gate,
            "w_q": model.w_q,
            "w_k": model.w_k,
            "w_v": model.w_v,
            "w_o": model.w_o,
            "regime_head": model.regime_head,
            "dir_head": model.dir_head,
        }
        np.savez(str(npz_path), **body_arrays)
        logger.info("Body weights saved to %s", npz_path)

        # Save sklearn classifiers
        clf_bundle = {
            "regime_clf": getattr(model, "_regime_clf", None),
            "dir_clf": getattr(model, "_dir_clf", None),
        }
        with open(pkl_path, "wb") as fh:
            pickle.dump(clf_bundle, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Classifiers saved to %s", pkl_path)

        # Save metadata
        meta = {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "hidden_dim": self.hidden_dim,
            "seq_len": SEQ_LEN,
            "n_features": N_FEATURES,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
        }
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)
        logger.info("Metadata saved to %s", meta_path)

    def _load_weights(self, symbol: str, timeframe: str) -> Optional[Any]:
        """
        Load saved weights if they exist.
        Returns a _NumpyTFT instance with weights restored, or None if not found.
        """
        return load_trained_model(symbol, timeframe)


# ---------------------------------------------------------------------------
# Sharpe proxy helper
# ---------------------------------------------------------------------------


def _compute_sharpe_proxy(preds: np.ndarray, y_true: np.ndarray) -> float:
    """
    Approximate Sharpe: assign +1 return for correct TREND call, -1 for wrong trend,
    0 for non-trend labels. Compute mean/std.
    """
    returns = []
    for p, t in zip(preds, y_true):
        if t in (0, 1):  # TREND_UP or TREND_DOWN
            returns.append(1.0 if p == t else -1.0)
        elif p == t:
            returns.append(0.1)  # small reward for correct non-trend
        else:
            returns.append(0.0)

    if not returns:
        return 0.0
    arr = np.array(returns)
    std = arr.std()
    if std < 1e-10:
        return 0.0
    return float(arr.mean() / std * math.sqrt(252))  # annualised


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------


def retrain_all_symbols(
    symbols: Optional[List[str]] = None,
    timeframe: str = "1h",
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """
    Retrain TFT for each symbol sequentially.

    Default symbols: BTC/USD, ETH/USD, SOL/USD.
    Returns dict of symbol -> metrics.
    """
    if symbols is None:
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]

    results: Dict[str, Dict[str, Any]] = {}
    for sym in symbols:
        logger.info("=== Retraining %s/%s ===", sym, timeframe)
        try:
            trainer = WalkForwardTrainer(symbol=sym, timeframe=timeframe, **kwargs)
            metrics = trainer.train()
            results[sym] = metrics
            logger.info(
                "Completed %s: OOS accuracy=%.3f, Sharpe proxy=%.3f",
                sym,
                float(metrics.get("oos_accuracy", 0.0)),
                float(metrics.get("oos_sharpe_proxy", 0.0)),
            )
        except Exception as exc:
            logger.exception("Failed to retrain %s: %s", sym, exc)
            results[sym] = {"error": str(exc)}

    return results


def load_trained_model(symbol: str, timeframe: str = "1h") -> Optional[Any]:
    """
    Load a trained _NumpyTFT model from ml/weights/.

    Returns a _NumpyTFT instance with body weights restored and sklearn
    classifiers attached as _regime_clf and _dir_clf, or None if not found.
    """
    if _NumpyTFT is None:
        logger.warning("_NumpyTFT not available; cannot load trained model")
        return None

    symbol_safe = symbol.replace("/", "_").replace(":", "_")
    stem = f"{symbol_safe}_{timeframe}"
    npz_path = WEIGHTS_DIR / f"{stem}.npz"
    pkl_path = WEIGHTS_DIR / f"{stem}_clf.pkl"
    meta_path = WEIGHTS_DIR / f"{stem}_meta.json"

    if not npz_path.exists():
        logger.info("No saved weights found at %s", npz_path)
        return None

    # Load body weights
    try:
        arrays = np.load(str(npz_path))
    except Exception as exc:
        logger.error("Failed to load body weights from %s: %s", npz_path, exc)
        return None

    # Determine hidden_dim from meta or infer from arrays
    hidden_dim = DEFAULT_HIDDEN
    if meta_path.exists():
        try:
            with open(meta_path) as fh:
                meta = json.load(fh)
            hidden_dim = int(meta.get("hidden_dim", DEFAULT_HIDDEN))
        except Exception as _e:
            logger.debug("train_tft error: %s", _e)

    model = _NumpyTFT(input_size=N_FEATURES, hidden=hidden_dim)

    # Restore body arrays
    for attr in ["vsn_w", "vsn_v", "grn_w1", "grn_w2", "grn_gate",
                 "w_q", "w_k", "w_v", "w_o", "regime_head", "dir_head"]:
        if attr in arrays:
            setattr(model, attr, arrays[attr])

    # Restore sklearn classifiers
    if pkl_path.exists():
        try:
            with open(pkl_path, "rb") as fh:
                clf_bundle = pickle.load(fh)
            model._regime_clf = clf_bundle.get("regime_clf")
            model._dir_clf = clf_bundle.get("dir_clf")
            logger.info("Classifiers loaded from %s", pkl_path)
        except Exception as exc:
            logger.warning("Could not load classifiers from %s: %s", pkl_path, exc)
            model._regime_clf = None
            model._dir_clf = None
    else:
        model._regime_clf = None
        model._dir_clf = None

    logger.info("Loaded trained model for %s/%s from %s", symbol, timeframe, npz_path)
    return model


# ---------------------------------------------------------------------------
# __main__ entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train TFT regime model")
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--months", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--oos-months", type=int, default=3)
    args = parser.parse_args()

    trainer = WalkForwardTrainer(
        symbol=args.symbol,
        timeframe=args.timeframe,
        train_months=args.months,
        oos_months=args.oos_months,
        epochs=args.epochs,
    )
    metrics = trainer.train()
    logger.info(json.dumps(metrics, indent=2, default=str))