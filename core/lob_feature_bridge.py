"""
core/lob_feature_bridge.py
==========================
Pipes LOBSnapshot data into FeatureStore.

Upgrade (2026-04 peak-potential):
- Micro-price feature: weighted midpoint using opposite-side queue depth
  (bid_px * ask_qty + ask_px * bid_qty) / (bid_qty + ask_qty). More
  accurate short-term price expectation than plain mid-price.
- Tiered queue imbalance: separate imbalance at levels 0, 1, 2 so the
  signal pipeline can detect level-specific order flow pressure.
- Rolling VWAP divergence: (current mid_price - rolling_vwap) / rolling_vwap
  over a configurable window of snapshots. Captures short-term price
  deviation from recent execution average.
- Stale snapshot guard: skips feature writes if ts_ns has not advanced
  since the last snapshot, preventing duplicate stale data poisoning the
  feature store.
- Async batch flush: buffers up to batch_size snapshots then writes all
  features in a single set_batch() call, reducing store contention under
  high-frequency feeds.

All original features preserved:
  lob_mid_price, lob_spread, lob_spread_bps, lob_order_imbalance,
  lob_bid/ask_depth, lob_depth_ratio, lob_bid/ask_px/qty_0..9,
  lob_vwap_bid, lob_vwap_ask, lob_ts_ns, lob_sequence
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

from core.feature_store import FeatureStore
from core.feeds.lob_feed import LOBSnapshot

logger = logging.getLogger("argus.core.lob_feature_bridge")

LOB_TTL_S: float = 10.0
VWAP_LEVELS: int = 5
TIERED_LEVELS: int = 3     # levels 0/1/2 for tiered imbalance
VWAP_WINDOW: int = 20      # rolling snap window for VWAP divergence


def extract_lob_features(
    snap: LOBSnapshot,
    rolling_mids: Optional[Deque[float]] = None,
) -> Dict[str, Any]:
    """Extract all LOB features from a snapshot.

    Parameters
    ----------
    snap : LOBSnapshot
    rolling_mids : deque of recent mid prices for VWAP divergence.
    """
    features: Dict[str, Any] = {}

    # ---- Core ----
    features["lob_mid_price"] = snap.mid_price
    features["lob_spread"] = snap.spread
    features["lob_spread_bps"] = (
        (snap.spread / snap.mid_price * 10_000) if snap.mid_price > 0 else 0.0
    )
    features["lob_order_imbalance"] = snap.order_imbalance
    features["lob_bid_depth"] = snap.bid_depth
    features["lob_ask_depth"] = snap.ask_depth
    features["lob_depth_ratio"] = (
        snap.bid_depth / snap.ask_depth if snap.ask_depth > 0 else 0.0
    )

    # ---- Level ladder ----
    for i in range(10):
        if i < len(snap.bids):
            features[f"lob_bid_px_{i}"] = snap.bids[i][0]
            features[f"lob_bid_qty_{i}"] = snap.bids[i][1]
        else:
            features[f"lob_bid_px_{i}"] = 0.0
            features[f"lob_bid_qty_{i}"] = 0.0
        if i < len(snap.asks):
            features[f"lob_ask_px_{i}"] = snap.asks[i][0]
            features[f"lob_ask_qty_{i}"] = snap.asks[i][1]
        else:
            features[f"lob_ask_px_{i}"] = 0.0
            features[f"lob_ask_qty_{i}"] = 0.0

    # ---- VWAP bid/ask (top 5 levels) ----
    bid_vwap_num = sum(
        snap.bids[i][0] * snap.bids[i][1]
        for i in range(min(VWAP_LEVELS, len(snap.bids)))
    )
    bid_vwap_den = sum(
        snap.bids[i][1] for i in range(min(VWAP_LEVELS, len(snap.bids)))
    )
    ask_vwap_num = sum(
        snap.asks[i][0] * snap.asks[i][1]
        for i in range(min(VWAP_LEVELS, len(snap.asks)))
    )
    ask_vwap_den = sum(
        snap.asks[i][1] for i in range(min(VWAP_LEVELS, len(snap.asks)))
    )
    features["lob_vwap_bid"] = bid_vwap_num / bid_vwap_den if bid_vwap_den > 0 else 0.0
    features["lob_vwap_ask"] = ask_vwap_num / ask_vwap_den if ask_vwap_den > 0 else 0.0

    # ---- Micro-price ----
    # Weighted midpoint: accounts for queue pressure at best bid/ask.
    if snap.bids and snap.asks:
        bp, bq = snap.bids[0]
        ap, aq = snap.asks[0]
        denom = bq + aq
        features["lob_micro_price"] = (
            (bp * aq + ap * bq) / denom if denom > 0 else snap.mid_price
        )
    else:
        features["lob_micro_price"] = snap.mid_price

    # ---- Tiered queue imbalance (levels 0, 1, 2) ----
    for lvl in range(TIERED_LEVELS):
        b_qty = snap.bids[lvl][1] if lvl < len(snap.bids) else 0.0
        a_qty = snap.asks[lvl][1] if lvl < len(snap.asks) else 0.0
        total = b_qty + a_qty
        features[f"lob_imbalance_l{lvl}"] = (
            (b_qty - a_qty) / total if total > 0 else 0.0
        )

    # ---- Rolling VWAP divergence ----
    if rolling_mids is not None and len(rolling_mids) > 0:
        rolling_vwap = sum(rolling_mids) / len(rolling_mids)
        features["lob_vwap_divergence"] = (
            (snap.mid_price - rolling_vwap) / rolling_vwap
            if rolling_vwap > 0 else 0.0
        )
    else:
        features["lob_vwap_divergence"] = 0.0

    features["lob_ts_ns"] = snap.ts_ns
    features["lob_sequence"] = snap.sequence
    return features


class LOBFeatureBridge:
    """
    Subscribes to LOB feed snapshots and writes features to FeatureStore.

    New behaviour:
    - Stale guard: skips write if ts_ns <= last ts_ns for the symbol.
    - Async batch flush: buffers up to batch_size snaps, flushes once.
    - Rolling mid deque: maintained per-symbol for VWAP divergence.
    """

    def __init__(
        self,
        feature_store: FeatureStore,
        ttl_s: float = LOB_TTL_S,
        log_every: int = 500,
        batch_size: int = 1,       # set >1 to enable batching
        vwap_window: int = VWAP_WINDOW,
    ) -> None:
        self._store = feature_store
        self._ttl = ttl_s
        self._log_every = log_every
        self._batch_size = max(1, int(batch_size))
        self._snap_count: int = 0
        self._last_log: float = 0.0
        # Stale guard: last ts_ns seen per symbol.
        self._last_ts_ns: Dict[str, int] = {}
        # Rolling mid deques per symbol.
        self._rolling_mids: Dict[str, Deque[float]] = {}
        self._vwap_window = int(vwap_window)
        # Batch buffer: symbol -> list of pending feature dicts.
        self._buffer: Dict[str, List[Dict[str, Any]]] = {}

    async def on_snapshot(self, snap: LOBSnapshot) -> None:
        self._ingest(snap)

    def on_snapshot_sync(self, snap: LOBSnapshot) -> None:
        self._ingest(snap)

    def _ingest(self, snap: LOBSnapshot) -> None:
        """Ingest a snapshot: stale check, feature extract, buffer/flush."""
        sym = snap.symbol

        # Stale guard.
        last_ts = self._last_ts_ns.get(sym, 0)
        if snap.ts_ns <= last_ts:
            logger.debug(
                "LOBFeatureBridge: stale snap skipped sym=%s ts_ns=%d last=%d",
                sym, snap.ts_ns, last_ts,
            )
            return
        self._last_ts_ns[sym] = snap.ts_ns

        # Update rolling mid deque.
        if sym not in self._rolling_mids:
            self._rolling_mids[sym] = collections.deque(maxlen=self._vwap_window)
        self._rolling_mids[sym].append(snap.mid_price)

        features = extract_lob_features(snap, rolling_mids=self._rolling_mids[sym])

        if self._batch_size == 1:
            self._store.set_batch(sym, features, ttl_s=self._ttl)
        else:
            self._buffer.setdefault(sym, []).append(features)
            if len(self._buffer[sym]) >= self._batch_size:
                self._flush(sym)

        self._snap_count += 1
        now = time.monotonic()
        if self._snap_count % self._log_every == 0 or now - self._last_log > 60:
            logger.debug(
                "LOBFeatureBridge: %s %d features written (snap #%d mid=%.4f imb=%.3f micro=%.4f)",
                sym, len(features), self._snap_count,
                snap.mid_price, snap.order_imbalance,
                features.get("lob_micro_price", snap.mid_price),
            )
            self._last_log = now

    def _flush(self, symbol: str) -> None:
        """Merge buffered feature dicts and write the latest values."""
        pending = self._buffer.pop(symbol, [])
        if not pending:
            return
        merged: Dict[str, Any] = {}
        for feat_dict in pending:
            merged.update(feat_dict)  # last snapshot wins per key
        self._store.set_batch(symbol, merged, ttl_s=self._ttl)

    def flush_all(self) -> None:
        """Force-flush all buffered symbols (call on shutdown)."""
        for sym in list(self._buffer.keys()):
            self._flush(sym)

    @property
    def stats(self) -> dict:
        return {
            "snapshots_processed": self._snap_count,
            "ttl_s": self._ttl,
            "batch_size": self._batch_size,
            "buffered_symbols": len(self._buffer),
            "tracked_symbols": len(self._last_ts_ns),
        }


def build_lob_pipeline(
    symbols: list,
    exchange: str = "kraken",
    feature_store: Optional[FeatureStore] = None,
    ttl_s: float = LOB_TTL_S,
    batch_size: int = 1,
) -> Tuple[list, LOBFeatureBridge]:
    """
    Build a complete LOB feed + bridge pipeline for a list of symbols.

    Default exchange is ``kraken`` (AU retail accessible).
    Set ``ARGUS_LOB_EXCHANGE=bybit`` or ``okx`` if outside AU.
    """
    from core.feeds.lob_feed import (
        KrakenLOBFeed, BinanceLOBFeed, BybitLOBFeed, OKXLOBFeed,
    )

    store = feature_store or FeatureStore(background=True, default_ttl_s=ttl_s)
    bridge = LOBFeatureBridge(store, ttl_s=ttl_s, batch_size=batch_size)

    feed_cls = {
        "kraken":  KrakenLOBFeed,
        "bybit":   BybitLOBFeed,
        "okx":     OKXLOBFeed,
        "binance": BinanceLOBFeed,
    }.get(exchange.lower(), KrakenLOBFeed)

    feeds = [
        feed_cls(sym, on_snapshot=bridge.on_snapshot)
        for sym in symbols
    ]

    logger.info(
        "build_lob_pipeline: %d %s feeds wired to LOBFeatureBridge (batch=%d)",
        len(feeds), exchange, batch_size,
    )
    return feeds, bridge
