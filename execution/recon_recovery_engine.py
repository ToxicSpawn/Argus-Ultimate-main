from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ReconRecoveryState:
    intent_id: str
    symbol: str
    state: str
    reason: str
    retry_count: int
    last_retry_ts: float
    recovery_status: str  # pending|retrying|cleared|halted
    resolution_reason: str


class ReconRecoveryEngine:
    """Deterministic recovery manager for stale RECON_REQUIRED intents."""

    def __init__(self, config: Any, state_store: Any) -> None:
        self.config = config
        self.state_store = state_store
        self.enabled = bool(getattr(config, "recon_recovery_enabled", True))
        self.stale_threshold_seconds = float(
            getattr(config, "recon_recovery_stale_threshold_seconds", 60.0) or 60.0
        )
        self.base_retry_delay_seconds = float(
            getattr(config, "recon_recovery_base_retry_delay_seconds", 5.0) or 5.0
        )
        self.max_retries = int(getattr(config, "recon_recovery_max_retries", 5) or 5)
        self.halt_on_retry_exhausted = bool(
            getattr(config, "recon_recovery_halt_on_retry_exhausted", True)
        )

    async def run_cycle(
        self,
        *,
        exchanges: Dict[str, Any],
        cycle_id: int,
        trace_id: str,
        reconcile_fn: Optional[Callable[..., Awaitable[Tuple[bool, Dict[str, Any]]]]] = None,
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "enabled": bool(self.enabled),
            "cycle_id": int(cycle_id),
            "trace_id": str(trace_id or ""),
            "stale_count": 0,
            "retried": 0,
            "cleared": 0,
            "halted": 0,
            "operator_review": 0,
            "classifications": {
                "auto_clearable": 0,
                "retryable": 0,
                "halt_required": 0,
                "operator_review": 0,
            },
            "halt_required": False,
            "by_symbol": {},
        }
        if not self.enabled or self.state_store is None:
            return summary

        now_ts = float(time.time())
        stale: List[Dict[str, Any]] = list(
            self.state_store.list_recon_required_intents(
                stale_after_seconds=float(self.stale_threshold_seconds),
                now_ts=now_ts,
            )
            or []
        )
        summary["stale_count"] = len(stale)
        if not stale:
            return summary

        pending_reconcile_retry = False
        stale_by_intent: Dict[str, Dict[str, Any]] = {str(r.get("intent_id") or ""): dict(r) for r in stale}

        for row in stale:
            intent_id = str(row.get("intent_id") or "")
            if not intent_id:
                continue
            # Skip reconciliation for paper/dry-run orders — they have no real exchange state
            order_id = str(row.get("exchange_order_id") or "")
            if order_id.startswith("dry_") or order_id.startswith("dry_ccxt_") or order_id.startswith("paper_"):
                # Auto-clear dry-run intents instead of retrying against exchange
                try:
                    self.state_store.update_intent_state(
                        intent_id,
                        "FAILED",
                        last_error="paper_order_auto_cleared",
                        details={"recon_recovery_cleared": True, "recon_recovery_resolution": "paper_order_auto_cleared"},
                    )
                except Exception as _e:
                    logger.debug("recon_recovery_engine error: %s", _e)
                self.state_store.upsert_recon_recovery_state(
                    intent_id=intent_id,
                    retry_count=0,
                    recovery_status="cleared",
                    last_retry_ts=float(time.time()),
                    resolution_reason="paper_order_auto_cleared",
                    recovery_classification="auto_clearable",
                )
                summary["cleared"] = int(summary.get("cleared", 0) or 0) + 1
                logger.debug("recon auto-cleared paper/dry-run intent %s (order_id=%s)", intent_id, order_id)
                continue
            symbol = str(row.get("symbol") or "")
            reason = str(row.get("last_error") or "")
            retry_row = self.state_store.get_recon_recovery_state(intent_id) or {}
            retry_count = int(retry_row.get("retry_count") or 0)
            last_retry_ts = float(retry_row.get("last_retry_ts") or 0.0)
            backoff = float(self.base_retry_delay_seconds) * (2 ** max(0, retry_count))
            due_retry = (last_retry_ts <= 0.0) or (now_ts - last_retry_ts >= backoff)

            if not due_retry:
                self._set_symbol_summary(
                    summary,
                    symbol=symbol,
                    status=str(retry_row.get("recovery_status") or "pending"),
                    retry_count=retry_count,
                    resolution=str(retry_row.get("resolution_reason") or ""),
                )
                continue

            classification, resolution_reason = await self._classify_intent(row=row, exchanges=exchanges)
            if classification not in {"auto_clearable", "retryable", "halt_required", "operator_review"}:
                classification = "operator_review"
            class_map = dict(summary.get("classifications", {}) or {})
            class_map[classification] = int(class_map.get(classification, 0) or 0) + 1
            summary["classifications"] = class_map

            if classification == "auto_clearable":
                try:
                    self.state_store.update_intent_state(
                        intent_id,
                        "FAILED",
                        last_error=str(resolution_reason),
                        details={
                            "recon_recovery_cleared": True,
                            "recon_recovery_cleared_ts": now_ts,
                            "recon_recovery_resolution": str(resolution_reason),
                        },
                    )
                except Exception as e:
                    logger.warning("recon recovery update_intent_state failed for %s: %s", intent_id, e)
                self.state_store.upsert_recon_recovery_state(
                    intent_id=intent_id,
                    retry_count=retry_count,
                    recovery_status="cleared",
                    last_retry_ts=now_ts,
                    resolution_reason=str(resolution_reason),
                    recovery_classification="auto_clearable",
                )
                summary["cleared"] = int(summary.get("cleared", 0) or 0) + 1
                self._set_symbol_summary(
                    summary,
                    symbol=symbol,
                    status="cleared",
                    retry_count=retry_count,
                    resolution=str(resolution_reason),
                    classification="auto_clearable",
                )
                logger.info("recon cleared intent %s for %s (%s)", intent_id, symbol or "UNKNOWN", resolution_reason)
                continue

            if classification == "operator_review":
                reason = str(resolution_reason or "operator_review_required")
                self.state_store.upsert_recon_recovery_state(
                    intent_id=intent_id,
                    retry_count=retry_count,
                    recovery_status="pending",
                    last_retry_ts=now_ts,
                    resolution_reason=reason,
                    recovery_classification="operator_review",
                )
                summary["operator_review"] = int(summary.get("operator_review", 0) or 0) + 1
                self._set_symbol_summary(
                    summary,
                    symbol=symbol,
                    status="pending",
                    retry_count=retry_count,
                    resolution=reason,
                    classification="operator_review",
                )
                logger.warning(
                    "recon operator review required for intent %s (%s): %s",
                    intent_id,
                    symbol or "UNKNOWN",
                    reason,
                )
                continue

            new_retry_count = retry_count + 1
            if new_retry_count > self.max_retries:
                if self.halt_on_retry_exhausted:
                    resolution = f"HALT_REQUIRED:max_retries_exhausted:{resolution_reason}"
                    status = "halted"
                    classification_now = "halt_required"
                    summary["halted"] = int(summary.get("halted", 0) or 0) + 1
                    class_map = dict(summary.get("classifications", {}) or {})
                    class_map["halt_required"] = int(class_map.get("halt_required", 0) or 0) + 1
                    summary["classifications"] = class_map
                else:
                    resolution = f"OPERATOR_REVIEW:max_retries_exhausted:{resolution_reason}"
                    status = "pending"
                    classification_now = "operator_review"
                    summary["operator_review"] = int(summary.get("operator_review", 0) or 0) + 1
                    class_map = dict(summary.get("classifications", {}) or {})
                    class_map["operator_review"] = int(class_map.get("operator_review", 0) or 0) + 1
                    summary["classifications"] = class_map
                self.state_store.upsert_recon_recovery_state(
                    intent_id=intent_id,
                    retry_count=new_retry_count,
                    recovery_status=status,
                    last_retry_ts=now_ts,
                    resolution_reason=resolution,
                    recovery_classification=classification_now,
                )
                self._set_symbol_summary(
                    summary,
                    symbol=symbol,
                    status=status,
                    retry_count=new_retry_count,
                    resolution=resolution,
                    classification=classification_now,
                )
                if classification_now == "halt_required":
                    logger.critical(
                        "recon escalation triggered for intent %s (%s): %s",
                        intent_id,
                        symbol or "UNKNOWN",
                        resolution,
                    )
                else:
                    logger.warning(
                        "recon retries exhausted; operator review required for intent %s (%s): %s",
                        intent_id,
                        symbol or "UNKNOWN",
                        resolution,
                    )
                continue

            self.state_store.upsert_recon_recovery_state(
                intent_id=intent_id,
                retry_count=new_retry_count,
                recovery_status="retrying",
                last_retry_ts=now_ts,
                resolution_reason=str(resolution_reason),
                recovery_classification="retryable",
            )
            summary["retried"] = int(summary.get("retried", 0) or 0) + 1
            self._set_symbol_summary(
                summary,
                symbol=symbol,
                status="retrying",
                retry_count=new_retry_count,
                resolution=str(resolution_reason),
                classification="retryable",
            )
            pending_reconcile_retry = True
            logger.warning(
                "recon retry triggered for intent %s (%s), retry_count=%s reason=%s",
                intent_id,
                symbol or "UNKNOWN",
                new_retry_count,
                resolution_reason,
            )

        if pending_reconcile_retry and reconcile_fn is not None:
            try:
                recon_ok, recon_payload = await reconcile_fn(cycle_id=int(cycle_id), trace_id=str(trace_id or ""))
            except Exception as e:
                recon_ok, recon_payload = False, {"error": str(e)}
            summary["reconcile_retry_ok"] = bool(recon_ok)
            summary["reconcile_retry_payload"] = dict(recon_payload or {})
            if recon_ok:
                for iid, row in stale_by_intent.items():
                    if not iid:
                        continue
                    if self.state_store.intent_is_recon_required(iid):
                        continue
                    cur = self.state_store.get_recon_recovery_state(iid) or {}
                    status_now = str(cur.get("recovery_status") or "")
                    if status_now == "cleared":
                        continue
                    self.state_store.upsert_recon_recovery_state(
                        intent_id=iid,
                        retry_count=int(cur.get("retry_count") or 0),
                        recovery_status="cleared",
                        last_retry_ts=now_ts,
                        resolution_reason="cleared_via_reconcile_retry",
                        recovery_classification="auto_clearable",
                    )
                    summary["cleared"] = int(summary.get("cleared", 0) or 0) + 1
                    self._set_symbol_summary(
                        summary,
                        symbol=str(row.get("symbol") or ""),
                        status="cleared",
                        retry_count=int(cur.get("retry_count") or 0),
                        resolution="cleared_via_reconcile_retry",
                        classification="auto_clearable",
                    )

        summary["halt_required"] = bool(int(summary.get("halted", 0) or 0) > 0)
        return summary

    async def _classify_intent(self, *, row: Dict[str, Any], exchanges: Dict[str, Any]) -> Tuple[str, str]:
        exchange_name = str(row.get("exchange") or "")
        symbol = str(row.get("symbol") or "")
        order_id = str(row.get("exchange_order_id") or "")
        logger.debug("_classify_intent: exchange=%s symbol=%s order_id=%s", exchange_name, symbol, order_id)

        exchange = exchanges.get(exchange_name)
        if exchange is None:
            logger.info("_classify_intent: exchange %r not available -> operator_review", exchange_name)
            return "operator_review", "exchange_unavailable"
        if not order_id:
            logger.info("_classify_intent: missing order_id for %s/%s -> operator_review", exchange_name, symbol)
            return "operator_review", "missing_exchange_order_id"
        fetch_order = getattr(exchange, "fetch_order", None)
        if not callable(fetch_order):
            logger.info("_classify_intent: exchange %r missing fetch_order -> operator_review", exchange_name)
            return "operator_review", "exchange_missing_fetch_order"

        try:
            raw = await fetch_order(order_id, symbol)
            logger.debug("_classify_intent: fetch_order %s returned status=%s", order_id, (raw or {}).get("status"))
        except Exception as e:
            msg = str(e).lower()
            if "not found" in msg or "unknown order" in msg:
                logger.info("_classify_intent: order %s not found on exchange -> auto_clearable", order_id)
                return "auto_clearable", "order_not_found"
            logger.warning("_classify_intent: exchange error for order %s: %s -> retryable", order_id, e)
            return "retryable", f"exchange_state_uncertain:{str(e)}"

        status = str((raw or {}).get("status") or "").lower()
        if status in {"canceled", "cancelled", "closed", "filled", "rejected", "expired"}:
            logger.info("_classify_intent: order %s terminal status=%s -> auto_clearable", order_id, status)
            return "auto_clearable", f"exchange_order_terminal:{status}"
        if status in {"open", "new", "partially_filled", "partial"}:
            logger.debug("_classify_intent: order %s active status=%s -> retryable", order_id, status)
            return "retryable", f"exchange_order_active:{status}"
        logger.warning("_classify_intent: order %s unknown status=%r -> operator_review", order_id, status)
        return "operator_review", "exchange_state_uncertain:unknown_status"

    @staticmethod
    def _set_symbol_summary(
        summary: Dict[str, Any],
        *,
        symbol: str,
        status: str,
        retry_count: int,
        resolution: str,
        classification: str = "",
    ) -> None:
        sym = str(symbol or "")
        if not sym:
            return
        summary.setdefault("by_symbol", {})
        summary["by_symbol"][sym] = {
            "recovery_status": str(status or "pending"),
            "retry_count": int(retry_count or 0),
            "resolution_reason": str(resolution or ""),
            "recovery_classification": str(classification or ""),
        }
