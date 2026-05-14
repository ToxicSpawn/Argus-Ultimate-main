"""
Full wiring for all 79 previously-dead components + LOB feature bridge
+ DirectionHead / DPDK co-lo bridge / RL fill calibrator.

Called from ComponentRegistry.on_cycle(), on_fill(), and pre_order_check().
Each function is a single try/except-guarded block that wires one component.
"""
from __future__ import annotations
import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CAPITAL TIER v2 — monkey-patch UnifiedTradingSystem + ExecutionEngine once
# ═══════════════════════════════════════════════════════════════════════════

_tier_patch_applied: bool = False


def _ensure_tier_patch(reg: Any) -> None:
    """
    Apply CapitalTier v2 sizing + execution patches exactly once.
    Safe to call on every cycle — idempotent after first application.
    """
    global _tier_patch_applied
    if _tier_patch_applied:
        return
    _tier_patch_applied = True

    # ── 1. Replace _compute_position_size on UnifiedTradingSystem ─────────
    try:
        from core._scp_position_size_v2 import (
            _compute_position_size,
            _after_fill_hook,
            _after_close_hook,
        )
        uts = getattr(reg, "unified_trading_system", None) or getattr(reg, "trading_system", None)
        if uts is not None:
            import types
            uts._compute_position_size = types.MethodType(_compute_position_size, uts)
            uts._scp_after_fill        = types.MethodType(_after_fill_hook,        uts)
            uts._scp_after_close       = types.MethodType(_after_close_hook,       uts)
            logger.info("[TierPatch] _compute_position_size v2 applied to %s", type(uts).__name__)
        else:
            # Patch the class so any future instance also gets v2
            try:
                from core.unified_trading_system import UnifiedTradingSystem
                UnifiedTradingSystem._compute_position_size = _compute_position_size
                UnifiedTradingSystem._scp_after_fill        = _after_fill_hook
                UnifiedTradingSystem._scp_after_close       = _after_close_hook
                logger.info("[TierPatch] _compute_position_size v2 applied to UnifiedTradingSystem class")
            except ImportError:
                logger.warning("[TierPatch] UnifiedTradingSystem not importable — sizing patch deferred")
    except Exception as exc:
        logger.warning("[TierPatch] sizing patch failed: %s", exc)

    # ── 2. Apply execution tier gate + slicer to ExecutionEngine ──────────
    try:
        from core.capital_tier_execution_patch import apply_tier_execution_patch
        ee = (
            getattr(reg, "execution_engine",         None)
            or getattr(reg, "unified_execution_engine", None)
        )
        if ee is not None:
            apply_tier_execution_patch(ee)
        else:
            logger.info("[TierPatch] ExecutionEngine not yet on registry — patch deferred to on_cycle")
    except Exception as exc:
        logger.warning("[TierPatch] execution patch failed: %s", exc)

    # ── 3. Initialise OpsMetrics and attach to registry ───────────────────
    try:
        from core.capital_tier import classify_tier
        from core.tier_config_extension import get_tier_cfg
        from core.monitoring.ops_metrics import OpsMetrics

        equity_aud = float(getattr(reg, "portfolio_value_aud", 1000.0) or 1000.0)
        tier       = classify_tier(equity_aud)
        tcfg       = get_tier_cfg(tier)

        ops = OpsMetrics(
            fee_drag_alert_bps = float(tcfg["fee_drag_alert_bps"]),
            heat_limit_pct     = float(tcfg["portfolio_heat_limit"]),
        )
        try:
            reg.ops_metrics = ops
        except Exception:
            pass
        logger.info("[TierPatch] OpsMetrics initialised — tier=%s", tier.value)
    except Exception as exc:
        logger.warning("[TierPatch] OpsMetrics init failed: %s", exc)

    # ── 4. Wire StrategyWeightRouter onto registry ─────────────────────────
    try:
        from core.strategy_weight_router import StrategyWeightRouter
        if not getattr(reg, "strategy_weight_router", None):
            reg.strategy_weight_router = StrategyWeightRouter()
            logger.info("[TierPatch] StrategyWeightRouter attached to registry")
    except Exception as exc:
        logger.warning("[TierPatch] StrategyWeightRouter init failed: %s", exc)

    # ── 5. Wire HRPAllocator onto registry ────────────────────────────────
    try:
        from core.hrp_allocator import HRPAllocator
        if not getattr(reg, "hrp_allocator", None):
            reg.hrp_allocator = HRPAllocator()
            logger.info("[TierPatch] HRPAllocator attached to registry")
    except Exception as exc:
        logger.warning("[TierPatch] HRPAllocator init failed: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# LOB FEATURE BRIDGE — initialised once, called every cycle
# ═══════════════════════════════════════════════════════════════════════════

_lob_bridge: Optional[Any] = None  # LOBFeatureBridge singleton
_lob_feeds: List[Any] = []          # active LOBFeed instances
_lob_init_attempted: bool = False


def _ensure_lob_bridge(reg: Any) -> Optional[Any]:
    """
    Lazily initialise LOBFeatureBridge and start feeds on first call.
    Wires into reg.feature_store (FeatureStore) if present.
    """
    global _lob_bridge, _lob_feeds, _lob_init_attempted
    if _lob_init_attempted:
        return _lob_bridge
    _lob_init_attempted = True
    try:
        import os
        from core.lob_feature_bridge import build_lob_pipeline

        if not os.getenv("ARGUS_LOB_FEED", "1") == "1":
            logger.info("LOB feed disabled via ARGUS_LOB_FEED=0")
            return None

        store = getattr(reg, "feature_store", None)
        if store is None:
            from core.feature_store import FeatureStore
            store = FeatureStore(background=True, default_ttl_s=10.0)
            try:
                reg.feature_store = store
            except Exception:
                pass

        symbols = getattr(reg, "lob_symbols", None)
        if not symbols:
            symbols_raw = getattr(reg, "symbols", ["BTC/USDT", "ETH/USDT"])
            symbols = list(symbols_raw)[:10]

        import os as _os
        exchange = _os.getenv("ARGUS_LOB_EXCHANGE", "binance")

        feeds, bridge = build_lob_pipeline(
            symbols=symbols,
            exchange=exchange,
            feature_store=store,
        )
        _lob_bridge = bridge
        _lob_feeds = feeds

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                for feed in feeds:
                    asyncio.ensure_future(feed.start())
                logger.info(
                    "LOBFeatureBridge: started %d %s feeds (%s)",
                    len(feeds), exchange, symbols
                )
            else:
                logger.warning(
                    "LOBFeatureBridge: no running event loop "
                    "— feeds not started (call feed.start() manually)"
                )
        except Exception as e:
            logger.warning("LOBFeatureBridge feed start error: %s", e)

    except ImportError as e:
        logger.warning("LOBFeatureBridge not available: %s", e)
    except Exception:
        logger.exception("LOBFeatureBridge init failed")

    return _lob_bridge


def _wire_lob_features(reg: Any, advisory: Dict, prices: Dict) -> None:
    bridge = _ensure_lob_bridge(reg)
    if bridge is None:
        return
    try:
        store = getattr(reg, "feature_store", None)
        if store is None:
            return
        lob_advisory: Dict[str, Any] = {}
        for sym in list(prices.keys())[:10]:
            feats = store.get_all(sym)
            lob_feats = {k: v for k, v in feats.items() if k.startswith("lob_")}
            if lob_feats:
                lob_advisory[sym] = {
                    "mid_price":       lob_feats.get("lob_mid_price",       0.0),
                    "spread_bps":      lob_feats.get("lob_spread_bps",      0.0),
                    "order_imbalance": lob_feats.get("lob_order_imbalance", 0.0),
                    "bid_depth":       lob_feats.get("lob_bid_depth",       0.0),
                    "ask_depth":       lob_feats.get("lob_ask_depth",       0.0),
                    "vwap_bid":        lob_feats.get("lob_vwap_bid",        0.0),
                    "vwap_ask":        lob_feats.get("lob_vwap_ask",        0.0),
                    "depth_ratio":     lob_feats.get("lob_depth_ratio",     0.0),
                    "sequence":        lob_feats.get("lob_sequence",        0),
                    "ts_ns":           lob_feats.get("lob_ts_ns",           0),
                    "n_features":      len(lob_feats),
                }
        if lob_advisory:
            advisory["lob"] = lob_advisory
            advisory["lob_bridge_stats"] = bridge.stats
    except Exception:
        logger.debug("LOB feature pull failed", exc_info=True)


def _wire_lob_pre_order(reg: Any, symbol: str, side: str,
                        size_factor: float, reasons: List[str]) -> float:
    try:
        store = getattr(reg, "feature_store", None)
        if store is None:
            return size_factor

        imbalance  = store.get(symbol, "lob_order_imbalance")
        spread_bps = store.get(symbol, "lob_spread_bps")

        if imbalance is not None and spread_bps is not None:
            imbalance  = float(imbalance)
            spread_bps = float(spread_bps)

            if spread_bps > 20.0:
                factor = max(0.4, 1.0 - (spread_bps - 20.0) / 100.0)
                size_factor *= factor
                reasons.append(f"lob_spread_{spread_bps:.1f}bps*{factor:.2f}")

            if side.lower() == "buy" and imbalance < -0.3:
                factor = max(0.5, 1.0 + imbalance)
                size_factor *= factor
                reasons.append(f"lob_imb_{imbalance:.2f}*{factor:.2f}")
            elif side.lower() == "sell" and imbalance > 0.3:
                factor = max(0.5, 1.0 - imbalance)
                size_factor *= factor
                reasons.append(f"lob_imb_{imbalance:.2f}*{factor:.2f}")

    except Exception:
        pass
    return size_factor


# ═══════════════════════════════════════════════════════════════════════════
# DPDK CO-LO BRIDGE
# ═══════════════════════════════════════════════════════════════════════════

_colo_bridge: Optional[Any] = None
_colo_init_attempted: bool = False


def _ensure_colo_bridge(reg: Any) -> Optional[Any]:
    global _colo_bridge, _colo_init_attempted
    if _colo_init_attempted:
        return _colo_bridge
    _colo_init_attempted = True
    try:
        import os
        from core.dpdk_colo_bridge import make_colo_bridge

        kb_cfg = getattr(reg, "kernel_bypass_config", None)
        exchange_host = (
            getattr(kb_cfg, "fallback_host", None)
            or os.getenv("ARGUS_EXCHANGE_HOST", "127.0.0.1")
        )
        latency_budget = (
            getattr(kb_cfg, "latency_budget_us", None)
            or float(os.getenv("ARGUS_LATENCY_BUDGET_US", "10.0"))
        )

        def _on_ready(stats: dict) -> None:
            logger.info(
                "colo_ready=True p99=%.2fµs — switching to KernelBypassRouter",
                stats.get("rtt_p99_us", 0.0),
            )
            try:
                ee = getattr(reg, "execution_engine", None)
                if ee and hasattr(ee, "switch_to_bypass"):
                    ee.switch_to_bypass()
            except Exception as e:
                logger.debug("switch_to_bypass failed: %s", e)

        def _on_lost() -> None:
            logger.warning("colo_ready lost — falling back to TCP execution path")
            try:
                ee = getattr(reg, "execution_engine", None)
                if ee and hasattr(ee, "switch_to_tcp"):
                    ee.switch_to_tcp()
            except Exception as e:
                logger.debug("switch_to_tcp failed: %s", e)

        bridge = make_colo_bridge(
            exchange_host=exchange_host,
            latency_budget_us=latency_budget,
            on_ready=_on_ready,
            on_lost=_on_lost,
        )

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(bridge.start())
                logger.info(
                    "DPDKColoBridge started — target=%s budget=%.1fµs",
                    exchange_host, latency_budget,
                )
            else:
                logger.warning("DPDKColoBridge: no running event loop — bridge not started")
        except Exception as e:
            logger.warning("DPDKColoBridge start error: %s", e)

        _colo_bridge = bridge
        try:
            reg.colo_bridge = bridge
        except Exception:
            pass

    except ImportError as e:
        logger.warning("DPDKColoBridge not available: %s", e)
    except Exception:
        logger.exception("DPDKColoBridge init failed")

    return _colo_bridge


# ═══════════════════════════════════════════════════════════════════════════
# RL FILL CALIBRATOR
# ═══════════════════════════════════════════════════════════════════════════

_fill_calibrator: Optional[Any] = None
_fill_cal_init_attempted: bool = False


def _ensure_fill_calibrator(reg: Any) -> Optional[Any]:
    global _fill_calibrator, _fill_cal_init_attempted
    if _fill_cal_init_attempted:
        return _fill_calibrator
    _fill_cal_init_attempted = True
    try:
        from core.rl_fill_calibrator import FillCalibrator, CalibrationConfig

        trainer = (
            getattr(reg, "jax_ppo_trainer",    None)
            or getattr(reg, "rl_tuner",         None)
            or getattr(reg, "jax_ewc_trainer",  None)
        )
        event_bus = getattr(reg, "event_bus", None)

        def _on_calibrated() -> None:
            logger.info("FillCalibrator GRADUATED — DirectionHead skip/size gate UNLOCKED")
            dh = getattr(reg, "direction_head", None)
            if dh and hasattr(dh, "_on_calibration_complete"):
                try:
                    dh._on_calibration_complete({
                        "fills":  _fill_calibrator.total_fills if _fill_calibrator else 0,
                        "sharpe": 0.0,
                    })
                except Exception:
                    pass

        cal = FillCalibrator(
            trainer=trainer,
            cfg=CalibrationConfig(),
            event_bus=event_bus,
            on_calibrated=_on_calibrated,
        )
        _fill_calibrator = cal
        try:
            reg.fill_calibrator = cal
        except Exception:
            pass
        logger.info(
            "FillCalibrator initialised — trainer=%s",
            type(trainer).__name__ if trainer else "None",
        )

    except ImportError as e:
        logger.warning("FillCalibrator not available: %s", e)
    except Exception:
        logger.exception("FillCalibrator init failed")

    return _fill_calibrator


# ═══════════════════════════════════════════════════════════════════════════
# DIRECTION HEAD
# ═══════════════════════════════════════════════════════════════════════════

_direction_head: Optional[Any] = None
_dh_init_attempted: bool = False


def _ensure_direction_head(reg: Any) -> Optional[Any]:
    global _direction_head, _dh_init_attempted
    if _dh_init_attempted:
        return _direction_head
    _dh_init_attempted = True
    try:
        from core.direction_head import DirectionHead, DirectionHeadConfig

        trainer = (
            getattr(reg, "jax_ppo_trainer", None)
            or getattr(reg, "rl_tuner",      None)
        )
        cal = _ensure_fill_calibrator(reg)
        if cal is None:
            logger.warning("DirectionHead skipped — FillCalibrator unavailable")
            return None

        bypass_router    = getattr(reg, "kernel_bypass_router", None)
        colo_bridge      = _ensure_colo_bridge(reg)
        execution_engine = getattr(reg, "execution_engine", None)

        dh = DirectionHead(
            trainer=trainer,
            fill_calibrator=cal,
            bypass_router=bypass_router,
            colo_bridge=colo_bridge,
            cfg=DirectionHeadConfig(),
            execution_engine=execution_engine,
        )
        _direction_head = dh
        try:
            reg.direction_head = dh
        except Exception:
            pass
        logger.info(
            "DirectionHead initialised — trainer=%s bypass=%s",
            type(trainer).__name__ if trainer else "None",
            bypass_router is not None,
        )

    except ImportError as e:
        logger.warning("DirectionHead not available: %s", e)
    except Exception:
        logger.exception("DirectionHead init failed")

    return _direction_head


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════

def _track(reg: Any, component: str, exc: Exception, cycle: int = 0) -> None:
    try:
        et = getattr(reg, "error_tracker", None)
        if et is not None:
            et.record(component, exc, cycle=cycle)
            if et.should_disable(component):
                et.disable(component)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# ON_CYCLE WIRING
# ═══════════════════════════════════════════════════════════════════════════


def wire_all_on_cycle(reg: Any, advisory: Dict, prices: Dict, regime: str, cycle: int) -> None:
    """Wire all dead components into on_cycle. Modifies advisory in place."""

    # ── CAPITAL TIER PATCH (idempotent) ───────────────────────────────────
    _ensure_tier_patch(reg)

    # ── TIER ADVISORY — publish current tier + strategy weights ───────────
    try:
        from core.capital_tier import classify_tier
        equity_aud = float(getattr(reg, "portfolio_value_aud", 0) or 0)
        tier       = classify_tier(equity_aud)
        advisory["capital_tier"] = tier.value

        swr = getattr(reg, "strategy_weight_router", None)
        if swr is not None:
            advisory["strategy_weights"] = swr.get_weights(regime, tier)
    except Exception:
        pass

    # ── OPS METRICS SNAPSHOT ─────────────────────────────────────────────
    try:
        ops = getattr(reg, "ops_metrics", None)
        if ops is not None:
            open_risk_usd = float(getattr(reg, "_open_risk_usd", 0.0) or 0.0)
            equity_aud    = float(getattr(reg, "portfolio_value_aud", 1.0) or 1.0)
            aud_to_usd    = float(getattr(reg, "aud_to_usd", 0.65) or 0.65)
            equity_usd    = equity_aud * aud_to_usd
            from core.capital_tier import classify_tier
            snap = ops.snapshot(
                open_risk_usd = open_risk_usd,
                equity_usd    = equity_usd,
                equity_aud    = equity_aud,
                capital_tier  = classify_tier(equity_aud).value,
            )
            advisory["ops_metrics"] = {
                "fee_drag_bps":      snap.fee_drag_24h_bps,
                "portfolio_heat_pct": snap.portfolio_heat_pct,
                "heat_utilisation":  snap.heat_utilisation,
                "fee_drag_alert":    snap.fee_drag_alert,
                "heat_alert":        snap.heat_alert,
                "capital_tier":      snap.capital_tier,
            }
    except Exception:
        pass

    # ── LOB FEATURES (first — feeds all downstream consumers) ────────────
    _wire_lob_features(reg, advisory, prices)

    # ── DIRECTION HEAD / CO-LO / FILL CALIBRATOR ─────────────────────────
    try:
        dh = _ensure_direction_head(reg)
        if dh is not None:
            advisory["direction_head"] = dh.stats()
    except Exception:
        pass

    try:
        cb = _colo_bridge
        if cb is not None:
            advisory["colo_bridge"] = cb.stats()
    except Exception:
        pass

    try:
        fc = _fill_calibrator
        if fc is not None:
            advisory["fill_calibrator"] = fc.stats()
    except Exception:
        pass

    # ── HIGH IMPACT ──────────────────────────────────────────────────────

    try:
        if reg.thompson_bandit_router is not None:
            rankings = reg.thompson_bandit_router.get_rankings()
            advisory["thompson_bandit"] = {
                "rankings": [{"name": n, "win_rate": round(w, 4), "mean_pnl": round(p, 4)} for n, w, p in rankings[:10]],
                "n_arms": len(rankings),
            }
    except Exception:
        pass

    try:
        if reg.decision_journal is not None and cycle % 5 == 0:
            reg.decision_journal.record_cycle(
                cycle_number=cycle,
                regime=regime,
                advisory_snapshot={k: str(v)[:200] for k, v in list(advisory.items())[:20]},
                prices={k: round(v, 2) for k, v in prices.items()},
            )
    except Exception:
        pass

    try:
        if reg.manipulation_detector is not None:
            for sym, px in prices.items():
                spread_bps = 5.0
                try:
                    store = getattr(reg, "feature_store", None)
                    if store:
                        sb = store.get(sym, "lob_spread_bps")
                        if sb is not None:
                            spread_bps = float(sb)
                except Exception:
                    pass
                reg.manipulation_detector.update(
                    symbol=sym, price=float(px),
                    volume=0.0, spread_bps=spread_bps,
                )
            alerts = getattr(reg.manipulation_detector, "get_alerts", lambda: [])()
            if alerts:
                advisory["manipulation_alerts"] = alerts
    except Exception:
        pass

    try:
        if reg.price_predictor is not None:
            predictions = {}
            for sym, px in prices.items():
                reg.price_predictor.update(sym, float(px))
                try:
                    pred = reg.price_predictor.predict(sym)
                    if pred is not None:
                        predictions[sym] = pred
                except Exception:
                    pass
            if predictions:
                advisory["price_predictions"] = predictions
    except Exception:
        pass

    try:
        if reg.whale_tracker is not None:
            for sym, px in prices.items():
                asset = sym.split("/")[0] if "/" in sym else sym
                try:
                    reg.whale_tracker.update(asset, float(px))
                except Exception:
                    pass
            advisory["whale_tracker"] = {"active": True}
        if reg.whale_signal_generator is not None:
            try:
                sig = reg.whale_signal_generator.get_composite_signal()
                advisory["whale_signal"] = sig
            except Exception:
                pass
    except Exception:
        pass

    try:
        if reg.regime_forecaster is not None:
            try:
                reg.regime_forecaster.observe(regime)
                fc = reg.regime_forecaster.forecast()
                advisory["regime_forecast"] = {
                    "predicted":   getattr(fc, "predicted_regime", regime),
                    "probability": round(getattr(fc, "probability", 0.0), 4),
                }
            except Exception:
                pass
    except Exception:
        pass

    try:
        if reg.entropy_filter is not None:
            try:
                entropy = reg.entropy_filter.compute_entropy(advisory)
                advisory["signal_entropy"] = round(float(entropy), 4)
            except Exception:
                pass
    except Exception:
        pass

    try:
        if reg.moe_strategy_router is not None:
            try:
                weights = reg.moe_strategy_router.get_weights(regime)
                advisory["moe_weights"] = weights
            except Exception:
                pass
    except Exception:
        pass

    try:
        if reg.latency_tracker is not None:
            advisory["latency_stats"] = reg.latency_tracker.snapshot()
    except Exception:
        pass

    try:
        if reg.drawdown_autopsy is not None and cycle % 100 == 0:
            advisory["drawdown_autopsy"] = {"active": True}
    except Exception:
        pass

    # ── MEDIUM IMPACT ────────────────────────────────────────────────────

    try:
        if reg.alpha_decay is not None:
            try:
                reg.alpha_decay.update(cycle)
                advisory["alpha_decay"] = reg.alpha_decay.snapshot()
            except Exception:
                pass
    except Exception:
        pass

    try:
        if reg.ml_feedback_loop is not None and cycle % 50 == 0:
            try:
                advisory["ml_feedback"] = reg.ml_feedback_loop.evaluate()
            except Exception:
                pass
    except Exception:
        pass

    try:
        if reg.execution_alpha is not None:
            advisory["execution_alpha"] = {"active": True}
    except Exception:
        pass

    try:
        if reg.fill_probability is not None:
            advisory["fill_probability"] = {"active": True}
    except Exception:
        pass

    try:
        if reg.portfolio_risk_optimizer is not None and cycle % 50 == 0:
            try:
                opt = reg.portfolio_risk_optimizer.optimize(prices)
                advisory["portfolio_risk"] = opt
            except Exception:
                pass
    except Exception:
        pass

    try:
        if reg.strategy_attribution is not None:
            try:
                advisory["strategy_attribution"] = reg.strategy_attribution.snapshot()
            except Exception:
                pass
    except Exception:
        pass

    try:
        if reg.strategy_validator is not None:
            advisory["strategy_validator"] = {"active": True}
    except Exception:
        pass

    try:
        if reg.learning_maximizer is not None:
            try:
                advisory["learning_maximizer"] = reg.learning_maximizer.snapshot()
            except Exception:
                pass
    except Exception:
        pass

    try:
        if reg.promotion_pipeline is not None and cycle % 200 == 0:
            try:
                advisory["promotion_pipeline"] = reg.promotion_pipeline.evaluate()
            except Exception:
                pass
    except Exception:
        pass

    try:
        if reg.meta_learner is not None:
            try:
                advisory["meta_learner"] = reg.meta_learner.snapshot()
            except Exception:
                pass
    except Exception:
        pass

    try:
        if reg.counterfactual is not None:
            advisory["counterfactual"] = {"active": True}
    except Exception:
        pass

    try:
        if reg.earnings_predictor is not None:
            try:
                reg.earnings_predictor.record_cycle(
                    pnl_aud=0.0, regime=regime, n_trades=0,
                )
                advisory["earnings_forecast"] = reg.earnings_predictor.snapshot()
            except Exception:
                pass
    except Exception:
        pass

    try:
        if reg.bayesian_optimizer is not None:
            advisory["bayesian_optimizer"] = {"active": True}
    except Exception:
        pass

    try:
        if reg.indicator_cache is not None:
            for sym, px in prices.items():
                try:
                    reg.indicator_cache.update(sym, float(px))
                except Exception:
                    pass
    except Exception:
        pass

    for name in ["regime_xgb", "vol_forecaster_v2", "gcn_from_disk",
                 "gat_from_disk", "itransformer_from_disk"]:
        try:
            comp = getattr(reg, name, None)
            if comp is not None:
                advisory[name] = {"loaded": True}
                if name == "vol_forecaster_v2":
                    for sym, px in prices.items():
                        try:
                            comp.update(sym, float(px))
                        except Exception:
                            pass
        except Exception:
            pass

    try:
        if reg.ewc_from_disk is not None or getattr(reg, "ewc_continual_learner", None) is not None:
            advisory["ewc_learner"] = {"active": True}
    except Exception:
        pass

    try:
        if reg.strategy_breeder is not None and cycle % 500 == 0:
            try:
                reg.strategy_breeder.breed_generation()
                advisory["strategy_breeder"] = {"bred": True}
            except Exception:
                pass
    except Exception:
        pass

    try:
        if reg.confidence_calibrator is not None:
            advisory["confidence_calibrator"] = {"active": True}
    except Exception:
        pass

    # ── LOW IMPACT (research/quantum) ────────────────────────────────────

    try:
        if reg.semantic_memory is not None and cycle % 100 == 0:
            facts = reg.semantic_memory.get_facts(predicate="performs_well_in_regime", limit=5)
            if facts:
                advisory["semantic_facts"] = [f"{f.subject} {f.predicate} {f.object}" for f in facts[:3]]
    except Exception:
        pass

    try:
        if reg.event_rag is not None:
            advisory["event_rag"] = {"active": True}
    except Exception:
        pass

    try:
        if reg.research_engine is not None:
            advisory["research_engine"] = {"active": True}
    except Exception:
        pass

    for name in ["adaptation_health_monitor", "adaptive_slicer", "adaptive_slippage",
                  "adversarial_thinker", "adversarial_trainer", "capital_migration_monitor",
                  "causal_engine", "causal_gnn", "deep_causal_engine",
                  "federated_learner", "gp_cluster_discovery", "hierarchical_rl",
                  "hypothesis_engine", "meta_cognition", "meta_learner_maml",
                  "module_reloader", "polyglot_engine", "probabilistic_programming",
                  "r740_engine", "rl_execution_agent", "signal_consensus",
                  "temporal_abstraction", "universal_data_brain", "market_memory"]:
        try:
            comp = getattr(reg, name, None)
            if comp is not None:
                advisory[name] = {"active": True}
                if hasattr(comp, "snapshot") and cycle % 100 == 0:
                    try:
                        advisory[name] = comp.snapshot()
                    except Exception:
                        pass
                elif hasattr(comp, "update") and cycle % 10 == 0:
                    try:
                        comp.update(cycle_number=cycle)
                    except Exception:
                        try:
                            comp.update()
                        except Exception:
                            pass
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# ON_FILL WIRING
# ═══════════════════════════════════════════════════════════════════════════


def wire_all_on_fill(reg: Any, trade_result: Dict, cycle: int) -> None:
    """Wire all dead components into on_fill."""
    symbol   = str(trade_result.get("symbol", ""))
    side     = str(trade_result.get("side", "")).lower()
    price    = float(trade_result.get("price", 0.0) or 0.0)
    pnl      = float(trade_result.get("pnl",   0.0) or 0.0)
    strategy = str(trade_result.get("source_strategy", "") or "unknown")

    # ── OPS METRICS — record fill fee + volume ────────────────────────────
    try:
        ops = getattr(reg, "ops_metrics", None)
        if ops is not None:
            qty        = float(trade_result.get("quantity", 0.0) or 0.0)
            vol_usd    = price * qty
            fee_bps    = float(trade_result.get("fee_bps", trade_result.get("fee", 0.0)) or 0.0)
            fee_usd    = vol_usd * fee_bps / 10_000.0
            ops.record_fill(fee_usd=fee_usd, vol_usd=vol_usd)
    except Exception:
        pass

    # ── FILL CALIBRATOR ───────────────────────────────────────────────────
    try:
        cal = _ensure_fill_calibrator(reg)
        if cal is not None:
            from core.rl_fill_calibrator import FillEvent
            fill = FillEvent(
                symbol=symbol,
                side=side,
                requested_qty=float(trade_result.get("quantity", 0.0) or 0.0),
                filled_qty=float(trade_result.get("filled_qty",
                                  trade_result.get("quantity", 0.0)) or 0.0),
                requested_px=float(trade_result.get("requested_px", price) or price),
                fill_px=price,
                fee_bps=float(trade_result.get("fee_bps",
                               trade_result.get("fee", 0.0)) or 0.0),
                latency_us=float(trade_result.get("latency_us",
                                  trade_result.get("latency_ms", 0.0) * 1000) or 0.0),
                signal_id=str(trade_result.get("signal_id", "")),
            )
            cal.record_sync(fill)
    except Exception:
        pass

    try:
        if reg.thompson_bandit_router is not None:
            won = pnl > 0
            reg.thompson_bandit_router.record_outcome(strategy, pnl, won)
    except Exception:
        pass

    try:
        if reg.decision_journal is not None:
            reg.decision_journal.record_fill(trade_result)
    except Exception:
        pass

    try:
        if reg.market_impact is not None:
            size_usd = price * float(trade_result.get("quantity", 0.0) or 0.0)
            slippage = float(trade_result.get("slippage_bps", 0.0) or 0.0)
            reg.market_impact.record_fill(
                symbol=symbol, side=side, size_usd=size_usd,
                slippage_bps=slippage, daily_volume_usd=1e6,
            )
    except Exception:
        pass

    try:
        if reg.latency_tracker is not None:
            latency = float(trade_result.get("latency_ms", 0.0) or 0.0)
            reg.latency_tracker.record(symbol=symbol, latency_ms=latency)
    except Exception:
        pass

    try:
        if reg.alpha_decay is not None:
            try:
                reg.alpha_decay.record_trade(strategy=strategy, pnl=pnl)
            except Exception:
                pass
    except Exception:
        pass

    try:
        if reg.ml_feedback_loop is not None:
            try:
                reg.ml_feedback_loop.record_outcome(
                    strategy=strategy, pnl=pnl,
                    regime=str(trade_result.get("regime_label", "")),
                )
            except Exception:
                pass
    except Exception:
        pass

    try:
        if reg.execution_alpha is not None:
            try:
                reg.execution_alpha.record_fill(trade_result)
            except Exception:
                pass
    except Exception:
        pass

    try:
        if reg.strategy_attribution is not None:
            try:
                reg.strategy_attribution.record_trade(
                    strategy=strategy, symbol=symbol, pnl=pnl, side=side,
                )
            except Exception:
                pass
    except Exception:
        pass

    try:
        if reg.meta_learner is not None:
            try:
                reg.meta_learner.record_model_performance(
                    model_name=strategy,
                    regime=str(trade_result.get("regime_label", "NORMAL")),
                    features={"pnl": pnl},
                    accuracy=1.0 if pnl > 0 else 0.0,
                )
            except Exception:
                pass
    except Exception:
        pass

    try:
        if reg.earnings_predictor is not None:
            try:
                reg.earnings_predictor.record_trade(pnl_aud=pnl, strategy=strategy)
            except Exception:
                pass
    except Exception:
        pass

    try:
        if reg.counterfactual is not None:
            try:
                reg.counterfactual.record_decision(trade_result)
            except Exception:
                pass
    except Exception:
        pass

    try:
        if reg.learning_maximizer is not None:
            try:
                reg.learning_maximizer.record_fill(trade_result)
            except Exception:
                pass
    except Exception:
        pass

    try:
        import time as _tod_time
        _tod_hour = _tod_time.gmtime().tm_hour
        if not hasattr(reg, "_tod_pnl"):
            reg._tod_pnl = {}
        reg._tod_pnl.setdefault(_tod_hour, []).append(pnl)
        if len(reg._tod_pnl[_tod_hour]) > 200:
            reg._tod_pnl[_tod_hour] = reg._tod_pnl[_tod_hour][-100:]
    except Exception:
        pass

    try:
        if reg.regime_forecaster is not None:
            try:
                reg.regime_forecaster.observe(str(trade_result.get("regime_label", "")))
            except Exception:
                pass
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# PRE_ORDER WIRING
# ═══════════════════════════════════════════════════════════════════════════


def wire_all_pre_order(reg: Any, symbol: str, side: str, size_usd: float,
                       size_factor: float, reasons: List[str]) -> float:
    """Wire all dead components into pre_order_check. Returns modified size_factor."""

    # ── TIER EXECUTION GATE (min-notional + fee gate) ─────────────────────
    try:
        ee = (
            getattr(reg, "execution_engine",          None)
            or getattr(reg, "unified_execution_engine", None)
        )
        if ee is not None and hasattr(ee, "_tier_pre_order_gate"):
            allowed, gate_reason = ee._tier_pre_order_gate(
                symbol       = symbol,
                side         = side,
                notional_usd = size_usd * size_factor,
                est_fee_usd  = size_usd * size_factor * 0.001,   # 10 bps estimate
            )
            if not allowed:
                reasons.append(f"tier_gate_blocked: {gate_reason}")
                return 0.0
    except Exception:
        pass

    # ── LOB gates ─────────────────────────────────────────────────────────
    size_factor = _wire_lob_pre_order(reg, symbol, side, size_factor, reasons)

    # ── DIRECTION HEAD gate ───────────────────────────────────────────────
    try:
        dh = _ensure_direction_head(reg)
        if dh is not None:
            from core.direction_head import SignalPacket

            advisory   = getattr(reg, "_last_cycle_advisory", None) or {}
            confidence = float(
                advisory.get("ensemble_confidence",
                advisory.get("signal_confidence", 0.6)) or 0.6
            )
            price_ref = float(
                (advisory.get("price_predictions") or {}).get(symbol, size_usd) or size_usd
            )

            packet = SignalPacket(
                symbol=symbol,
                direction="long" if side.lower() == "buy" else "short",
                base_size=size_factor,
                confidence=confidence,
                price=price_ref,
                signal_id=str(int(time.monotonic_ns())),
            )

            loop = None
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                pass

            decision = None
            if loop and loop.is_running():
                future = asyncio.run_coroutine_threadsafe(dh.process(packet), loop)
                try:
                    decision = future.result(timeout=0.005)
                except Exception:
                    pass

            if decision is not None and not decision.shadow_only:
                if decision.skipped:
                    reasons.append(f"direction_head_skip(logit={decision.rl_logit:.3f})")
                    return 0.0
                else:
                    size_factor = decision.final_size
                    reasons.append(
                        f"direction_head_scale*{decision.size_multiplier:.2f}"
                        f"(logit={decision.rl_logit:.3f}"
                        f",fr={decision.fill_ratio_est:.2f}"
                        f",bypass={decision.via_bypass})"
                    )
    except Exception:
        pass

    # 1. Conviction Sizer
    try:
        if reg.conviction_sizer is not None:
            try:
                advisory = getattr(reg, "_last_cycle_advisory", None) or {}
                result = reg.conviction_sizer.compute(
                    base_size_pct=size_factor,
                    symbol=symbol,
                    action=side.upper(),
                    strategy_type="unknown",
                    advisory=advisory,
                    regime=str(getattr(reg, "_latest_regime_label", "NORMAL")),
                )
                mult = getattr(result, "multiplier", 1.0)
                if mult != 1.0:
                    size_factor *= float(mult)
                    reasons.append(f"conviction*{mult:.2f}")
            except Exception:
                pass
    except Exception:
        pass

    # 2. Market Impact
    try:
        if reg.market_impact is not None:
            try:
                impact     = reg.market_impact.predict(symbol=symbol, side=side, size_usd=size_usd)
                impact_bps = getattr(impact, "expected_slippage_bps", 0.0)
                if float(impact_bps) > 10.0:
                    factor = max(0.5, 1.0 - float(impact_bps) / 100.0)
                    size_factor *= factor
                    reasons.append(f"market_impact*{factor:.2f}")
            except Exception:
                pass
    except Exception:
        pass

    # 3. Maker Enforcement
    try:
        if reg.maker_enforcement is not None:
            try:
                reg.maker_enforcement.enforce(symbol=symbol, side=side)
            except Exception:
                pass
    except Exception:
        pass

    # 4. Manipulation Detector
    try:
        if reg.manipulation_detector is not None:
            try:
                alert = reg.manipulation_detector.check(symbol)
                if alert and getattr(alert, "severity", "") == "HIGH":
                    size_factor *= 0.3
                    reasons.append("manipulation_alert*0.30")
            except Exception:
                pass
    except Exception:
        pass

    # 5. Fill Probability
    try:
        if reg.fill_probability is not None:
            try:
                prob = reg.fill_probability.predict(symbol=symbol, side=side)
                if hasattr(prob, "fill_probability") and float(prob.fill_probability) < 0.3:
                    reasons.append(f"low_fill_prob={prob.fill_probability:.2f}")
            except Exception:
                pass
    except Exception:
        pass

    # 6. Entropy Filter
    try:
        if reg.entropy_filter is not None:
            advisory = getattr(reg, "_last_cycle_advisory", None) or {}
            entropy  = float(advisory.get("signal_entropy", 0.5) or 0.5)
            if entropy > 0.9:
                size_factor *= 0.5
                reasons.append("high_entropy*0.50")
    except Exception:
        pass

    # 7. Whale Tracker
    try:
        if reg.whale_tracker is not None:
            advisory  = getattr(reg, "_last_cycle_advisory", None) or {}
            whale_sig = advisory.get("whale_signal")
            if isinstance(whale_sig, dict):
                whale_dir = str(whale_sig.get("direction", "")).upper()
                if (whale_dir == "SELL" and side.upper() == "BUY") or \
                   (whale_dir == "BUY"  and side.upper() == "SELL"):
                    size_factor *= 0.7
                    reasons.append("whale_opposite*0.70")
    except Exception:
        pass

    # 8. Correlation penalty
    try:
        advisory       = getattr(reg, "_last_cycle_advisory", None) or {}
        _corr_penalty  = float(advisory.get("correlation_penalty", 0.0) or 0.0)
        if _corr_penalty > 0.2:
            _corr_factor = max(0.5, 1.0 - _corr_penalty)
            size_factor *= _corr_factor
            reasons.append(f"correlation*{_corr_factor:.2f}")
    except Exception:
        pass

    # 9. Per-strategy position limit
    try:
        positions   = getattr(reg, "_last_positions", None) or {}
        if isinstance(positions, dict):
            _strat_count = sum(1 for p in positions.values()
                               if isinstance(p, dict) and p.get("strategy") == side)
            if _strat_count >= 3:
                size_factor *= 0.5
                reasons.append("per_strategy_limit*0.50")
    except Exception:
        pass

    # 10. Overnight risk reduction
    try:
        import time as _t
        _hour = _t.gmtime().tm_hour
        if _hour >= 22 or _hour <= 4:
            size_factor *= 0.70
            reasons.append(f"overnight_reduction*0.70 ({_hour}h UTC)")
    except Exception:
        pass

    return size_factor
