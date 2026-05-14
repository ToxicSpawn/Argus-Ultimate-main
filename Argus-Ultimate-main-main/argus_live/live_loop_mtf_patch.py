"""
Argus Live — MTF Patch for the Main Trading Loop.

Drop-in patch that adds MTF confluence gating to any Argus
UnifiedTradingSystem (UTS) or argus_live entrypoint without modifying
the core UTS class.

How to activate
---------------
Add the following near the top of ``run_ultimate.py`` or the argus_live
entrypoint *after* exchanges and config are constructed but *before*
the main loop starts::

    from argus_live.live_loop_mtf_patch import MTFPatch

    mtf_patch = MTFPatch.build(
        config=config,
        exchanges=exchanges,
        dry_run=False,          # set True for shadow-mode first
    )
    asyncio.create_task(mtf_patch.start())  # starts WS + seed in background

Then, inside the trading cycle (typically in
``UnifiedTradingSystem._trading_cycle()`` or wherever signals are
collected before ``execute_signals()``)::

    signals = await self._collect_signals()   # existing call
    signals = mtf_patch.gate(signals)         # <-- insert this line
    results = await self.execution_engine.execute_signals(signals)

Configuration (all optional, read from config or passed to build())::

    config.mtf_timeframes        = ["15m", "1h", "4h"]   # default
    config.mtf_min_agreeing_tfs  = 2                      # default
    config.mtf_min_score         = 0.6                    # default
    config.mtf_dry_run           = False                  # default
    config.mtf_seed_candles      = 100                    # default
    config.mtf_fast_ema          = 9
    config.mtf_slow_ema          = 21
    config.mtf_trend_ema         = 50
    config.mtf_require_higher_tf = True
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MTFPatch:
    """
    Wraps MTFBarCloseRouter + SignalGate into a single object
    for convenient injection into the live trading loop.

    Attributes:
        router:     MTFBarCloseRouter instance.
        gate:       SignalGate instance (use ``gate.filter_signals()``).
        mtf_filter: Underlying MTFConfluenceFilter.
    """

    def __init__(self, router: Any, gate: Any, mtf_filter: Any):
        self.router = router
        self.gate = gate
        self.mtf_filter = mtf_filter
        self._task: Optional[asyncio.Task] = None

    @classmethod
    def build(
        cls,
        config: Any,
        exchanges: Dict[str, Any],
        jsonl_logger: Optional[Any] = None,
        dry_run: Optional[bool] = None,
    ) -> "MTFPatch":
        """
        Factory: build MTFPatch from config + exchanges dict.

        Reads all MTF settings from config attributes with safe defaults.

        Args:
            config:       Argus config object.
            exchanges:    Dict of exchange_name -> ccxt instance.
            jsonl_logger: Optional JSONLLogger for audit trail.
            dry_run:      Override dry_run mode (else from config).

        Returns:
            Ready-to-use MTFPatch (call ``.start()`` to begin feeding).
        """
        from strategies.mtf_confluence import MTFConfluenceFilter
        from argus_live.mtf_bar_router import MTFBarCloseRouter
        from argus_live.signal_gate import SignalGate

        tfs = list(getattr(config, "mtf_timeframes", None) or ["15m", "1h", "4h"])
        min_agreeing = int(getattr(config, "mtf_min_agreeing_tfs", 2) or 2)
        min_score_cfg = float(getattr(config, "mtf_min_score", 0.6) or 0.6)
        dry_run_cfg = bool(dry_run if dry_run is not None else getattr(config, "mtf_dry_run", False))
        seed_candles = int(getattr(config, "mtf_seed_candles", 100) or 100)
        fast_ema = int(getattr(config, "mtf_fast_ema", 9) or 9)
        slow_ema = int(getattr(config, "mtf_slow_ema", 21) or 21)
        trend_ema = int(getattr(config, "mtf_trend_ema", 50) or 50)
        require_higher = bool(getattr(config, "mtf_require_higher_tf", True))
        symbols = list(
            getattr(config, "trading_pairs", None)
            or getattr(config, "symbols", None)
            or []
        )

        mtf_filter = MTFConfluenceFilter(
            timeframes=tfs,
            min_agreeing_tfs=min_agreeing,
            min_confluence_score=min_score_cfg,
            fast_ema=fast_ema,
            slow_ema=slow_ema,
            trend_ema=trend_ema,
            require_higher_tf_agreement=require_higher,
            enforce_closed_candles=True,
        )

        router = MTFBarCloseRouter(
            mtf_filter=mtf_filter,
            exchanges=exchanges,
            symbols=symbols,
            timeframes=tfs,
            seed_candles=seed_candles,
        )

        gate = SignalGate(
            mtf_filter=mtf_filter,
            dry_run=dry_run_cfg,
            min_score=min_score_cfg,
            jsonl_logger=jsonl_logger,
        )

        logger.info(
            "MTFPatch built: tfs=%s min_agreeing=%d min_score=%.2f dry_run=%s symbols=%s",
            tfs, min_agreeing, min_score_cfg, dry_run_cfg, symbols,
        )
        return cls(router=router, gate=gate, mtf_filter=mtf_filter)

    async def start(self) -> None:
        """
        Start the MTFBarCloseRouter in the background.

        Call as ``asyncio.create_task(mtf_patch.start())``.
        Completes when router.run() finishes (i.e. never in normal operation).
        """
        logger.info("MTFPatch: starting bar-close router")
        await self.router.run()

    def gate_signals(self, signals: List[Any], *, run_id: str = "", cycle_id: int = 0) -> List[Any]:
        """
        Synchronously gate a list of signals through MTF confluence.

        This is the single line you insert before ``execute_signals()``.

        Args:
            signals:   Candidate signal list.
            run_id:    Current run ID for audit.
            cycle_id:  Current cycle ID for audit.

        Returns:
            Filtered signal list (blocked signals removed).
        """
        return self.gate.filter_signals(signals, run_id=run_id, cycle_id=cycle_id)

    def get_gate_stats(self) -> Dict[str, Any]:
        """Return gate statistics for Prometheus / dashboard."""
        return self.gate.get_stats()

    def get_filter_analysis(self, symbol: str) -> Dict[str, Any]:
        """Return per-TF analysis for a symbol (for logging/dashboard)."""
        return self.mtf_filter.get_timeframe_analysis(symbol)
