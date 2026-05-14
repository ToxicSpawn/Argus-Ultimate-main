from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _getf(d: Dict[str, Any], k: str, default: float = 0.0) -> float:
    try:
        return float(d.get(k, default) or default)
    except Exception as e:
        logger.debug("_getf conversion failed for key=%s: %s", k, e)
        return float(default)


@dataclass(frozen=True)
class PromotionDecision:
    ok: bool
    reason: str
    details: Dict[str, Any]


def score_result(r: Dict[str, Any]) -> float:
    """
    Risk-adjusted score used for promotion comparisons.
    """
    sharpe = _getf(r, "sharpe", 0.0)
    sortino = _getf(r, "sortino", 0.0)
    maxdd = _getf(r, "max_drawdown_pct", 0.0)
    return float(sharpe + 0.5 * sortino - 0.10 * maxdd)


class PromotionGate:
    """
    Decide whether to promote a candidate configuration to live.
    """

    def __init__(
        self,
        *,
        min_delta_score: float = 0.10,
        max_drawdown_pct: float = 10.0,
        min_trades: int = 3,
        require_all_timeframes: bool = True,
    ) -> None:
        self.min_delta_score = float(min_delta_score)
        self.max_drawdown_pct = float(max_drawdown_pct)
        self.min_trades = int(min_trades)
        self.require_all_timeframes = bool(require_all_timeframes)

    def evaluate(
        self,
        *,
        baseline_by_tf: Dict[str, Dict[str, Any]],
        candidate_by_tf: Dict[str, Dict[str, Any]],
        timeframes: List[str],
    ) -> PromotionDecision:
        tfs = [str(t) for t in timeframes if str(t).strip()]
        if not tfs:
            return PromotionDecision(ok=False, reason="no_timeframes", details={})

        missing = [t for t in tfs if t not in baseline_by_tf or t not in candidate_by_tf]
        if missing and self.require_all_timeframes:
            return PromotionDecision(ok=False, reason="missing_timeframes", details={"missing": missing})

        deltas: List[Tuple[str, float]] = []
        checks: Dict[str, Any] = {}
        ok_any = False

        for tf in tfs:
            b = baseline_by_tf.get(tf) or {}
            c = candidate_by_tf.get(tf) or {}
            if not b or not c:
                continue

            b_tr = int(_getf(b, "trades", 0.0))
            c_tr = int(_getf(c, "trades", 0.0))
            b_dd = _getf(b, "max_drawdown_pct", 0.0)
            c_dd = _getf(c, "max_drawdown_pct", 0.0)
            if c_tr < self.min_trades:
                checks[tf] = {"ok": False, "reason": "min_trades", "candidate_trades": c_tr}
                continue
            if c_dd > self.max_drawdown_pct:
                checks[tf] = {"ok": False, "reason": "max_drawdown", "candidate_maxdd": c_dd}
                continue

            b_s = score_result(b)
            c_s = score_result(c)
            d = float(c_s - b_s)
            deltas.append((tf, d))
            checks[tf] = {
                "ok": d >= self.min_delta_score,
                "baseline_score": b_s,
                "candidate_score": c_s,
                "delta_score": d,
                "baseline_trades": b_tr,
                "candidate_trades": c_tr,
                "baseline_maxdd_pct": b_dd,
                "candidate_maxdd_pct": c_dd,
            }
            if d >= self.min_delta_score:
                ok_any = True

        if self.require_all_timeframes:
            ok_all = all(bool((checks.get(tf) or {}).get("ok")) for tf in tfs if tf in checks)
            return PromotionDecision(ok=bool(ok_all), reason="ok_all" if ok_all else "failed_some", details=checks)

        return PromotionDecision(ok=bool(ok_any), reason="ok_any" if ok_any else "failed_all", details=checks)

