"""
MicroStrategyOrchestrator — Top-level coordinator for $1k AUD HFT bot.

Orchestrates:
  - MicroCapitalMM: altcoin market making on Bybit (zero maker fee)
  - FundingRateArb: delta-neutral carry trade on Bybit perpetuals
  - MicroCapitalAllocator: hourly capital rebalancing
  - MicroRiskEnvelope: pre-trade risk checks and daily PnL management

Default safe mode: paper_trading=True — real money requires explicit opt-in.

Performance projections (conservative):
  MM (3 pairs, 50bps avg spread, 40 fills/day):
    ~$0.80/day gross → $292/yr → ~47% on $620
  Funding arb ($248 allocated, 15% ann. yield):
    ~$0.10/day        → $37/yr
  Combined gross: ~$0.90/day ($329/yr, ~53%)
  Realistic after adverse selection: ~$0.40-0.60/day ($146-$219/yr, ~24-35%)

Usage::

    config = OrchestratorConfig(
        total_capital_aud=1000.0,
        paper_trading=True,
    )
    orch = MicroStrategyOrchestrator(config)
    await orch.start()
    # ... run until stop signal ...
    await orch.stop()
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.micro_capital_allocator import (
    Allocation,
    AllocatorConfig,
    MicroCapitalAllocator,
)
from core.micro_risk_envelope import MicroRiskEnvelope, MicroRiskConfig

# Strategy imports — conditionally imported to allow unit testing without
# exchange clients available.
try:
    from strategies.micro_capital_mm import MicroCapitalMM, MicroMMConfig
    _MM_AVAILABLE = True
except ImportError:
    _MM_AVAILABLE = False
    MicroCapitalMM = None  # type: ignore[assignment,misc]
    MicroMMConfig = None   # type: ignore[assignment,misc]

try:
    from strategies.funding_rate_arb import FundingRateArb, FundingArbConfig
    _FUNDING_AVAILABLE = True
except ImportError:
    _FUNDING_AVAILABLE = False
    FundingRateArb = None  # type: ignore[assignment,misc]
    FundingArbConfig = None  # type: ignore[assignment,misc]

logger = logging.getLogger("argus.micro_orchestrator")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NS_PER_SECOND: int = 1_000_000_000
AUD_USD_RATE_DEFAULT: float = 0.62
SECONDS_PER_HOUR: float = 3600.0
HOURS_PER_YEAR: float = 8760.0

# Conservative performance projections for dashboard
_PROJ_MM_DAILY_USD: float = 0.80         # gross MM income per day
_PROJ_FUNDING_DAILY_USD: float = 0.10    # gross funding arb per day
_PROJ_NET_DAILY_USD: float = 0.50        # realistic after losses


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorConfig:
    """Top-level configuration for MicroStrategyOrchestrator."""
    total_capital_aud: float = 1000.0
    aud_usd_rate: float = AUD_USD_RATE_DEFAULT   # will be auto-fetched if possible
    mm_enabled: bool = True
    funding_enabled: bool = True
    paper_trading: bool = True              # always default True for safety
    rebalance_interval_s: float = 3600.0   # hourly
    stats_log_interval_s: float = 3600.0   # hourly stats logging

    # Exchange connection configs — dict format mirrors existing project conventions
    exchange_configs: Dict[str, Any] = field(default_factory=lambda: {
        "bybit": {
            "api_key": "",
            "api_secret": "",
            "testnet": True,     # default to testnet
        },
        "kraken": {
            "api_key": "",
            "api_secret": "",
        },
        "coinbase": {
            "api_key": "",
            "api_secret": "",
        },
    })

    @property
    def total_capital_usd(self) -> float:
        return self.total_capital_aud * self.aud_usd_rate

    def __post_init__(self) -> None:
        if self.total_capital_aud <= 0:
            raise ValueError("total_capital_aud must be positive")
        if self.aud_usd_rate <= 0 or self.aud_usd_rate > 5:
            raise ValueError("aud_usd_rate seems invalid — check exchange rate")


# ---------------------------------------------------------------------------
# MicroStrategyOrchestrator
# ---------------------------------------------------------------------------

class MicroStrategyOrchestrator:
    """
    Top-level orchestrator for the $1,000 AUD (~$620 USD) micro-capital HFT bot.

    Responsibilities:
      1. Initialise and configure all sub-systems
      2. Distribute capital based on MicroCapitalAllocator decisions
      3. Run hourly rebalance loop
      4. Monitor global risk via MicroRiskEnvelope
      5. Emergency halt on killswitch trigger
      6. Log daily performance summaries
      7. Provide dashboard and performance report APIs

    All operations are async-first. Use asyncio.run(orch.start()) or integrate
    into an existing event loop.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None) -> None:
        self._cfg = config or OrchestratorConfig()
        self._start_time_ns: int = 0
        self._running: bool = False
        self._halted: bool = False

        # Sub-system instances (initialised in start())
        self._allocator: Optional[MicroCapitalAllocator] = None
        self._risk: Optional[MicroRiskEnvelope] = None
        self._mm: Optional[Any] = None          # MicroCapitalMM instance
        self._funding: Optional[Any] = None     # FundingRateArb instance

        # Accumulated session PnL (used for dashboard when strategies unavailable)
        self._session_mm_pnl: float = 0.0
        self._session_funding_pnl: float = 0.0

        # Background tasks
        self._rebalance_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None

        logger.info(
            "MicroStrategyOrchestrator created: AUD=$%.2f USD=$%.2f paper=%s",
            self._cfg.total_capital_aud,
            self._cfg.total_capital_usd,
            self._cfg.paper_trading,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Initialise all sub-systems and start background loops.

        Sequence:
          1. Try to fetch live AUD/USD rate (fallback to config default)
          2. Build AllocatorConfig + MicroRiskConfig from OrchestratorConfig
          3. Instantiate MicroCapitalAllocator and get initial allocation
          4. Instantiate MicroRiskEnvelope
          5. Instantiate MicroCapitalMM (if mm_enabled and available)
          6. Instantiate FundingRateArb (if funding_enabled and available)
          7. Start rebalance loop
          8. Start daily stats logger
        """
        if self._running:
            logger.warning("Orchestrator already running — ignoring start() call")
            return

        self._start_time_ns = time.time_ns()
        self._running = True

        # Step 1: refresh AUD/USD rate
        await self._refresh_aud_usd_rate()

        total_usd = self._cfg.total_capital_usd
        logger.info(
            "Starting orchestrator: $%.2f AUD = $%.2f USD (rate=%.4f) paper=%s",
            self._cfg.total_capital_aud,
            total_usd,
            self._cfg.aud_usd_rate,
            self._cfg.paper_trading,
        )

        # Step 2: build sub-system configs
        alloc_config = AllocatorConfig(
            total_capital_usd=total_usd,
            rebalance_interval_s=self._cfg.rebalance_interval_s,
        )
        risk_config = MicroRiskConfig(
            total_capital_usd=total_usd,
        )

        # Step 3: allocator
        self._allocator = MicroCapitalAllocator(alloc_config)
        allocation = self._allocator.get_allocation()
        logger.info(
            "Initial allocation: MM=$%.2f Funding=$%.2f Reserve=$%.2f",
            allocation.mm_capital_usd,
            allocation.funding_capital_usd,
            allocation.reserve_usd,
        )

        # Step 4: risk envelope
        self._risk = MicroRiskEnvelope(risk_config)

        # Step 5: MM strategy
        if self._cfg.mm_enabled and _MM_AVAILABLE and MicroMMConfig is not None:
            mm_config = MicroMMConfig(
                total_capital_usd=allocation.mm_capital_usd,
            )
            self._mm = MicroCapitalMM(mm_config)
            logger.info(
                "MicroCapitalMM initialised with $%.2f capital",
                allocation.mm_capital_usd,
            )
        elif self._cfg.mm_enabled:
            logger.warning(
                "MM enabled but MicroCapitalMM not importable — running without MM"
            )

        # Step 6: funding arb strategy
        if self._cfg.funding_enabled and _FUNDING_AVAILABLE and FundingArbConfig is not None:
            funding_config = FundingArbConfig(
                capital_usd=allocation.funding_capital_usd,
            )
            self._funding = FundingRateArb(funding_config)
            logger.info(
                "FundingRateArb initialised with $%.2f capital",
                allocation.funding_capital_usd,
            )
        elif self._cfg.funding_enabled:
            logger.warning(
                "Funding enabled but FundingRateArb not importable — running without funding"
            )

        # Step 7: start background loops
        self._rebalance_task = asyncio.create_task(
            self._rebalance_loop(), name="orchestrator_rebalance"
        )
        self._stats_task = asyncio.create_task(
            self._stats_logger_loop(), name="orchestrator_stats"
        )

        logger.info("Orchestrator started — all sub-systems active")

    async def stop(self) -> None:
        """
        Gracefully shut down all strategies and log final session summary.

        Sequence:
          1. Signal running=False
          2. Cancel all MM orders
          3. Close all funding positions
          4. Cancel background tasks
          5. Log final session report
        """
        if not self._running:
            return

        self._running = False
        logger.info("Orchestrator stopping...")

        # Cancel MM orders
        if self._mm is not None:
            try:
                await self._mm.stop()
                logger.info("MM orders cancelled")
            except Exception as exc:  # pragma: no cover
                logger.error("Error stopping MM: %s", exc)

        # Close funding positions
        if self._funding is not None:
            try:
                await self._funding.close_all()
                logger.info("Funding positions closed")
            except Exception as exc:  # pragma: no cover
                logger.error("Error closing funding positions: %s", exc)

        # Cancel background tasks
        for task in (self._rebalance_task, self._stats_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Final summary
        report = self.get_performance_report()
        logger.info("=== FINAL SESSION REPORT ===\n%s", report)

    async def rebalance_loop(self) -> None:
        """
        Public-facing rebalance loop — delegates to internal implementation.
        Called every rebalance_interval_s to adjust strategy capital.
        """
        await self._rebalance_loop()

    def get_dashboard(self) -> Dict[str, Any]:
        """
        Return a comprehensive real-time dashboard dict.

        Keys:
            total_capital_aud, total_capital_usd
            mm_status, mm_pnl, mm_pairs, mm_allocation
            funding_status, funding_pnl, funding_positions, funding_allocation
            global_pnl_usd, global_pnl_aud, drawdown_pct, risk_status
            uptime_hours, estimated_daily_yield_usd, estimated_annual_yield_pct
        """
        now_ns = time.time_ns()
        uptime_s = (now_ns - self._start_time_ns) / NS_PER_SECOND if self._start_time_ns else 0
        uptime_hours = uptime_s / SECONDS_PER_HOUR

        total_usd = self._cfg.total_capital_usd
        total_aud = self._cfg.total_capital_aud

        # MM status
        mm_allocation = 0.0
        mm_pnl = self._session_mm_pnl
        mm_pairs: List[str] = []
        mm_status = "disabled"

        if self._allocator is not None:
            alloc = self._allocator.get_allocation()
            mm_allocation = alloc.mm_capital_usd
        if self._mm is not None:
            try:
                mm_stat = self._mm.get_status()
                mm_status = "running" if self._running and not self._halted else "halted"
                mm_pnl = mm_stat.get("session_pnl_usd", self._session_mm_pnl)
                mm_pairs = list(mm_stat.get("active_pairs", {}).keys())
            except Exception:  # pragma: no cover
                mm_status = "error"
        elif self._cfg.mm_enabled:
            mm_status = "paused"

        # Funding status
        funding_allocation = 0.0
        funding_pnl = self._session_funding_pnl
        funding_positions: List[str] = []
        funding_status = "disabled"

        if self._allocator is not None:
            alloc = self._allocator.get_allocation()
            funding_allocation = alloc.funding_capital_usd
        if self._funding is not None:
            try:
                yield_info = self._funding.get_total_yield()
                funding_status = "running" if self._running and not self._halted else "halted"
                funding_pnl = yield_info.get("total_realised_usd", self._session_funding_pnl)
                funding_positions = [
                    p.symbol for p in self._funding.get_active_positions()
                ]
            except Exception:  # pragma: no cover
                funding_status = "error"
        elif self._cfg.funding_enabled:
            funding_status = "paused"

        # Global PnL and risk
        global_pnl_usd = mm_pnl + funding_pnl
        global_pnl_aud = global_pnl_usd / self._cfg.aud_usd_rate
        drawdown_pct = abs(global_pnl_usd) / total_usd * 100 if global_pnl_usd < 0 else 0.0

        risk_status_dict: Dict[str, Any] = {"status": "green"}
        if self._risk is not None:
            try:
                risk_status_dict = self._risk.get_risk_status()
            except Exception:  # pragma: no cover
                pass

        # Estimated yield projections
        estimated_daily_yield_usd = _PROJ_NET_DAILY_USD
        estimated_annual_yield_usd = estimated_daily_yield_usd * 365
        estimated_annual_yield_pct = estimated_annual_yield_usd / total_usd * 100

        return {
            # Capital
            "total_capital_aud": round(total_aud, 2),
            "total_capital_usd": round(total_usd, 2),
            "aud_usd_rate": self._cfg.aud_usd_rate,
            "paper_trading": self._cfg.paper_trading,
            # MM
            "mm_status": mm_status,
            "mm_pnl": round(mm_pnl, 4),
            "mm_pairs": mm_pairs,
            "mm_allocation": round(mm_allocation, 2),
            # Funding
            "funding_status": funding_status,
            "funding_pnl": round(funding_pnl, 4),
            "funding_positions": funding_positions,
            "funding_allocation": round(funding_allocation, 2),
            # Global PnL
            "global_pnl_usd": round(global_pnl_usd, 4),
            "global_pnl_aud": round(global_pnl_aud, 4),
            "drawdown_pct": round(drawdown_pct, 4),
            "risk_status": risk_status_dict.get("status", "unknown"),
            # Uptime and projections
            "uptime_hours": round(uptime_hours, 2),
            "estimated_daily_yield_usd": round(estimated_daily_yield_usd, 4),
            "estimated_annual_yield_pct": round(estimated_annual_yield_pct, 2),
            # Allocator stats
            "allocator_stats": self._allocator.get_stats() if self._allocator else {},
        }

    def get_performance_report(self) -> str:
        """
        Generate a human-readable daily performance report.

        Returns:
            Formatted multi-line string with session summary.
        """
        dash = self.get_dashboard()
        rate = self._cfg.aud_usd_rate

        lines = [
            "=" * 60,
            "   ARGUS MICRO-CAPITAL TRADING REPORT",
            "=" * 60,
            f"  Capital:       AUD ${dash['total_capital_aud']:,.2f}  "
            f"(USD ${dash['total_capital_usd']:,.2f})",
            f"  Mode:          {'PAPER TRADING' if dash['paper_trading'] else 'LIVE'}",
            f"  Uptime:        {dash['uptime_hours']:.2f} hours",
            "",
            "  ── MARKET MAKING ───────────────────────────────",
            f"  Status:        {dash['mm_status'].upper()}",
            f"  Allocation:    USD ${dash['mm_allocation']:,.2f}",
            f"  Session PnL:   USD ${dash['mm_pnl']:+.4f}  "
            f"(AUD ${dash['mm_pnl'] / rate:+.4f})",
            f"  Active pairs:  {', '.join(dash['mm_pairs']) if dash['mm_pairs'] else 'none'}",
            "",
            "  ── FUNDING RATE ARB ─────────────────────────────",
            f"  Status:        {dash['funding_status'].upper()}",
            f"  Allocation:    USD ${dash['funding_allocation']:,.2f}",
            f"  Session PnL:   USD ${dash['funding_pnl']:+.4f}  "
            f"(AUD ${dash['funding_pnl'] / rate:+.4f})",
            f"  Open positions: {', '.join(dash['funding_positions']) if dash['funding_positions'] else 'none'}",
            "",
            "  ── GLOBAL SUMMARY ───────────────────────────────",
            f"  Total PnL:     USD ${dash['global_pnl_usd']:+.4f}  "
            f"(AUD ${dash['global_pnl_aud']:+.4f})",
            f"  Drawdown:      {dash['drawdown_pct']:.2f}% of capital",
            f"  Risk status:   {dash['risk_status'].upper()}",
            "",
            "  ── PROJECTIONS (CONSERVATIVE) ───────────────────",
            f"  Est. daily:    USD ${dash['estimated_daily_yield_usd']:.2f}/day",
            f"  Est. annual:   {dash['estimated_annual_yield_pct']:.1f}% p.a. on ${dash['total_capital_usd']:,.0f}",
            "  Note: Projections assume 50bps avg spread, 40 fills/day on 3 pairs.",
            "        Realistic range: 24–35% after adverse selection losses.",
            "=" * 60,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Background loops
    # ------------------------------------------------------------------

    async def _rebalance_loop(self) -> None:
        """
        Hourly rebalance loop.

        Each iteration:
          1. Collect performance metrics from active strategies
          2. Feed into allocator
          3. Call allocator.rebalance() → new Allocation
          4. If allocation changed materially: resize strategy capital
          5. If global killswitch triggered: halt everything
        """
        logger.info(
            "Rebalance loop started (interval=%.0fs)", self._cfg.rebalance_interval_s
        )

        while self._running:
            try:
                await asyncio.sleep(self._cfg.rebalance_interval_s)

                if not self._running:
                    break

                await self._perform_rebalance()

            except asyncio.CancelledError:
                break
            except Exception as exc:  # pragma: no cover
                logger.error("Rebalance loop error: %s", exc, exc_info=True)
                # Don't crash the loop on transient errors
                await asyncio.sleep(60.0)

        logger.info("Rebalance loop exited")

    async def _perform_rebalance(self) -> None:
        """Execute a single rebalance cycle."""
        if self._allocator is None or self._risk is None:
            return

        now_ns = time.time_ns()

        # Collect performance from MM
        mm_pnl_delta = 0.0
        mm_sharpe = 0.0
        if self._mm is not None:
            try:
                mm_stat = self._mm.get_status()
                mm_pnl_delta = mm_stat.get("last_hour_pnl_usd", 0.0)
                mm_sharpe = mm_stat.get("sharpe_ratio", 0.0)
                self._session_mm_pnl = mm_stat.get("session_pnl_usd", self._session_mm_pnl)
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not get MM stats: %s", exc)

        # Collect performance from funding arb
        funding_pnl_delta = 0.0
        funding_sharpe = 0.0
        if self._funding is not None:
            try:
                yield_info = self._funding.get_total_yield()
                funding_pnl_delta = yield_info.get("last_hour_pnl_usd", 0.0)
                ann_yield = yield_info.get("annualised_yield", 0.0)
                # Convert annualised yield to a Sharpe-like metric for allocator
                funding_sharpe = ann_yield / 0.15 if ann_yield > 0 else 0.0
                self._session_funding_pnl = yield_info.get(
                    "total_realised_usd", self._session_funding_pnl
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not get funding stats: %s", exc)

        # Feed performance into allocator
        self._allocator.update_performance(
            strategy="mm",
            pnl=mm_pnl_delta,
            sharpe=mm_sharpe,
            timestamp_ns=now_ns,
        )
        self._allocator.update_performance(
            strategy="funding",
            pnl=funding_pnl_delta,
            sharpe=funding_sharpe,
            timestamp_ns=now_ns,
        )

        # Check global killswitch
        if self._allocator.check_global_killswitch():
            logger.critical(
                "GLOBAL KILLSWITCH TRIGGERED — halting all strategies"
            )
            self._halted = True
            await self._emergency_halt()
            return

        # Run rebalance
        old_alloc = self._allocator.get_allocation()
        new_alloc = self._allocator.rebalance()

        # Detect material change (>$5 shift)
        mm_changed = abs(new_alloc.mm_capital_usd - old_alloc.mm_capital_usd) > 5.0
        funding_changed = abs(
            new_alloc.funding_capital_usd - old_alloc.funding_capital_usd
        ) > 5.0

        if mm_changed or funding_changed:
            logger.info(
                "Capital rebalanced: MM $%.2f→$%.2f | Funding $%.2f→$%.2f | %s",
                old_alloc.mm_capital_usd,
                new_alloc.mm_capital_usd,
                old_alloc.funding_capital_usd,
                new_alloc.funding_capital_usd,
                new_alloc.reason,
            )
            await self._apply_allocation(new_alloc)

    async def _apply_allocation(self, alloc: Allocation) -> None:
        """
        Push new capital allocations to the active strategies.
        Strategies resize their order quantities accordingly.
        """
        # MM: update capital — MicroCapitalMM exposes config.total_capital_usd
        if self._mm is not None and hasattr(self._mm, "_config"):
            try:
                self._mm._config.total_capital_usd = alloc.mm_capital_usd
                if alloc.mm_capital_usd <= 0:
                    await self._mm.stop()
                    logger.info("MM stopped: capital allocation = $0")
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not update MM capital: %s", exc)

        # Funding: update capital
        if self._funding is not None and hasattr(self._funding, "_config"):
            try:
                self._funding._config.capital_usd = alloc.funding_capital_usd
                if alloc.funding_capital_usd <= 0:
                    await self._funding.close_all()
                    logger.info("Funding positions closed: capital allocation = $0")
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not update funding capital: %s", exc)

    async def _emergency_halt(self) -> None:
        """Emergency halt — cancel all orders and close all positions immediately."""
        logger.critical("EMERGENCY HALT: cancelling all orders and closing positions")

        if self._mm is not None:
            try:
                await self._mm.stop()
            except Exception as exc:  # pragma: no cover
                logger.error("Emergency halt MM stop error: %s", exc)

        if self._funding is not None:
            try:
                await self._funding.close_all()
            except Exception as exc:  # pragma: no cover
                logger.error("Emergency halt funding close error: %s", exc)

        self._running = False

    async def _stats_logger_loop(self) -> None:
        """Hourly stats logger — writes structured performance data to logs."""
        while self._running:
            try:
                await asyncio.sleep(self._cfg.stats_log_interval_s)

                if not self._running:
                    break

                report = self.get_performance_report()
                logger.info("Hourly stats:\n%s", report)

            except asyncio.CancelledError:
                break
            except Exception as exc:  # pragma: no cover
                logger.error("Stats logger error: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # AUD/USD rate fetch
    # ------------------------------------------------------------------

    async def _refresh_aud_usd_rate(self) -> None:
        """
        Attempt to fetch live AUD/USD rate.
        Falls back silently to config default on any error.
        """
        try:
            # Try to get rate from Bybit ticker if client is available
            # Format: AUDUSD direct or via USDT-AUD proxy
            # We use a simple HTTP approach as a best-effort
            import aiohttp
            url = "https://api.bybit.com/v5/market/tickers?category=spot&symbol=AUDUSD"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5.0)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        result = data.get("result", {}).get("list", [])
                        if result:
                            rate = float(result[0].get("lastPrice", 0))
                            if 0.4 < rate < 1.2:  # sanity check for AUD/USD
                                self._cfg.aud_usd_rate = rate
                                logger.info("AUD/USD rate refreshed: %.4f", rate)
                                return
        except Exception as exc:
            logger.debug("AUD/USD rate fetch failed: %s — using default %.4f", exc, self._cfg.aud_usd_rate)

        logger.info("Using default AUD/USD rate: %.4f", self._cfg.aud_usd_rate)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MicroStrategyOrchestrator("
            f"aud=${self._cfg.total_capital_aud:.0f}, "
            f"usd=${self._cfg.total_capital_usd:.0f}, "
            f"paper={self._cfg.paper_trading}, "
            f"running={self._running}, "
            f"halted={self._halted})"
        )
