"""
Exchange Startup Validator — pre-trade connectivity and readiness checks.

Runs before the trading loop starts in live mode to verify:
  1. API key validity (authenticated endpoint test)
  2. Sufficient balances
  3. Trading pair availability
  4. Rate limit headroom

If any critical check fails, live trading is blocked.

Usage:
    from core.exchange_validator import ExchangeValidator
    validator = ExchangeValidator(exchanges)
    report = await validator.run_all(min_balance_usd=50.0, pairs=["BTC/USD"])
    if not report["all_passed"]:
        sys.exit(1)
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ExchangeValidator:
    """
    Pre-trade validation for exchange connectivity and readiness.

    Args:
        exchanges: Dict mapping exchange name to ccxt async exchange instance
                   or ExchangeConnector-like object.
    """

    def __init__(self, exchanges: Optional[Dict[str, Any]] = None):
        self._exchanges = exchanges or {}

    # ------------------------------------------------------------------ public

    async def run_all(
        self,
        min_balance_usd: float = 10.0,
        pairs: Optional[List[str]] = None,
        timeout: float = 15.0,
    ) -> Dict[str, Any]:
        """
        Run all validation checks and return a consolidated report.

        Args:
            min_balance_usd: Minimum total USD-equivalent balance required.
            pairs:           Trading pairs to validate (e.g. ["BTC/USD"]).
            timeout:         Seconds to allow for each exchange check.

        Returns:
            {
                "all_passed": bool,
                "exchanges": {
                    "kraken": {"status": "ok"|"fail", "issues": [...]},
                    ...
                }
            }
        """
        pairs = pairs or []
        report: Dict[str, Any] = {"all_passed": True, "exchanges": {}}

        for name, ex in self._exchanges.items():
            ex_report: Dict[str, Any] = {"status": "ok", "issues": []}

            # 1. API key validation
            key_result = await self._check_with_timeout(
                self.validate_api_keys(name, ex), timeout, "api_key_check"
            )
            if not key_result.get("ok", False):
                ex_report["issues"].append(
                    f"API key validation failed: {key_result.get('error', 'unknown')}"
                )

            # 2. Balance check
            balance_result = await self._check_with_timeout(
                self.validate_balances(name, ex, min_balance_usd), timeout, "balance_check"
            )
            if not balance_result.get("ok", False):
                ex_report["issues"].append(
                    f"Balance check failed: {balance_result.get('error', 'unknown')}"
                )
            if balance_result.get("total_usd", 0) > 0:
                ex_report["total_usd"] = balance_result["total_usd"]

            # 3. Trading pair validation
            if pairs:
                pair_result = await self._check_with_timeout(
                    self.validate_trading_pairs(name, ex, pairs), timeout, "pair_check"
                )
                if not pair_result.get("ok", False):
                    ex_report["issues"].append(
                        f"Pair validation failed: {pair_result.get('error', 'unknown')}"
                    )
                if pair_result.get("unavailable"):
                    ex_report["issues"].append(
                        f"Unavailable pairs: {pair_result['unavailable']}"
                    )

            # 4. Rate limit check
            rate_result = await self._check_with_timeout(
                self.validate_rate_limits(name, ex), timeout, "rate_limit_check"
            )
            if not rate_result.get("ok", False):
                ex_report["issues"].append(
                    f"Rate limit check: {rate_result.get('error', 'info not available')}"
                )
            if rate_result.get("latency_ms"):
                ex_report["latency_ms"] = rate_result["latency_ms"]

            if ex_report["issues"]:
                ex_report["status"] = "fail"
                report["all_passed"] = False

            report["exchanges"][name] = ex_report

        return report

    # ------------------------------------------------------------------ individual checks

    async def validate_api_keys(self, name: str, ex: Any) -> Dict[str, Any]:
        """
        Test API key validity by calling a private endpoint (fetch_balance).

        Returns:
            {"ok": bool, "error": str|None}
        """
        try:
            if hasattr(ex, "fetch_balance"):
                await ex.fetch_balance()
                return {"ok": True}
            elif hasattr(ex, "get_balance"):
                await ex.get_balance()
                return {"ok": True}
            else:
                return {"ok": False, "error": "no balance endpoint available"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    async def validate_balances(
        self, name: str, ex: Any, min_balance_usd: float
    ) -> Dict[str, Any]:
        """
        Check that the exchange has at least min_balance_usd total value.

        Returns:
            {"ok": bool, "total_usd": float, "error": str|None}
        """
        try:
            if hasattr(ex, "fetch_balance"):
                balance = await ex.fetch_balance()
            elif hasattr(ex, "get_balance"):
                balance = await ex.get_balance()
            else:
                return {"ok": False, "total_usd": 0, "error": "no balance endpoint"}

            # CCXT balance format: {currency: {free, used, total}, ...}
            total_usd = 0.0
            if isinstance(balance, dict):
                # Check for 'total' key (CCXT standard)
                total_section = balance.get("total", balance)
                if isinstance(total_section, dict):
                    for currency, amount in total_section.items():
                        if currency in ("info", "free", "used", "total", "timestamp", "datetime"):
                            continue
                        val = float(amount or 0)
                        if val <= 0:
                            continue
                        # Rough USD conversion for common currencies
                        if currency in ("USD", "USDT", "USDC", "BUSD", "DAI"):
                            total_usd += val
                        elif currency == "AUD":
                            total_usd += val * 0.65  # approximate
                        elif currency == "BTC":
                            total_usd += val * 60000  # approximate
                        elif currency == "ETH":
                            total_usd += val * 3000  # approximate
                        else:
                            # Skip unknown currencies
                            pass

            ok = total_usd >= min_balance_usd
            result: Dict[str, Any] = {"ok": ok, "total_usd": total_usd}
            if not ok:
                result["error"] = f"insufficient balance: ${total_usd:.2f} < ${min_balance_usd:.2f}"
            return result
        except Exception as exc:
            return {"ok": False, "total_usd": 0, "error": str(exc)}

    async def validate_trading_pairs(
        self, name: str, ex: Any, pairs: List[str]
    ) -> Dict[str, Any]:
        """
        Verify that the requested trading pairs are available on the exchange.

        Returns:
            {"ok": bool, "available": [...], "unavailable": [...]}
        """
        try:
            available = []
            unavailable = []

            if hasattr(ex, "load_markets"):
                await ex.load_markets()
                markets = getattr(ex, "markets", {}) or {}
                for pair in pairs:
                    if pair in markets:
                        available.append(pair)
                    else:
                        unavailable.append(pair)
            elif hasattr(ex, "fetch_ticker"):
                # Test each pair individually
                for pair in pairs:
                    try:
                        ticker = await ex.fetch_ticker(pair)
                        if ticker:
                            available.append(pair)
                        else:
                            unavailable.append(pair)
                    except Exception:
                        unavailable.append(pair)
            else:
                return {"ok": False, "available": [], "unavailable": pairs, "error": "no market data endpoint"}

            ok = len(unavailable) == 0
            result: Dict[str, Any] = {"ok": ok, "available": available, "unavailable": unavailable}
            if unavailable:
                result["error"] = f"{len(unavailable)} pairs unavailable"
            return result
        except Exception as exc:
            return {"ok": False, "available": [], "unavailable": pairs, "error": str(exc)}

    async def validate_rate_limits(self, name: str, ex: Any) -> Dict[str, Any]:
        """
        Check API rate limit status by measuring a lightweight request latency.

        Returns:
            {"ok": bool, "latency_ms": float}
        """
        try:
            t0 = time.perf_counter()
            if hasattr(ex, "fetch_ticker"):
                await ex.fetch_ticker("BTC/USD")
            elif hasattr(ex, "get_ticker"):
                await ex.get_ticker("BTC/USD")
            else:
                return {"ok": False, "error": "no ticker endpoint"}
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # Consider degraded if latency > 5 seconds
            ok = latency_ms < 5000.0
            result: Dict[str, Any] = {"ok": ok, "latency_ms": round(latency_ms, 1)}
            if not ok:
                result["error"] = f"high latency: {latency_ms:.0f}ms"
            return result
        except Exception as exc:
            return {"ok": False, "latency_ms": 0, "error": str(exc)}

    # ------------------------------------------------------------------ helpers

    async def _check_with_timeout(
        self, coro: Any, timeout: float, label: str
    ) -> Dict[str, Any]:
        """Run a check coroutine with a timeout."""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            return {"ok": False, "error": f"{label} timed out after {timeout}s"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


async def validate_exchange_startup(
    exchanges: Dict[str, Any],
    min_balance_usd: float = 10.0,
    pairs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function: run all validations and return the report.

    Args:
        exchanges:       Dict of exchange name -> ccxt exchange instance.
        min_balance_usd: Minimum required balance.
        pairs:           Trading pairs to check.

    Returns:
        Validation report dict.
    """
    validator = ExchangeValidator(exchanges)
    return await validator.run_all(
        min_balance_usd=min_balance_usd,
        pairs=pairs,
    )
