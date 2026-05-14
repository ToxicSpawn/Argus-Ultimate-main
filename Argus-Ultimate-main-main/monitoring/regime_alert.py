#!/usr/bin/env python3
"""
Regime Change Alerter — sends Discord webhook notifications on market regime
transitions detected by the trading system.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class RegimeChangeAlerter:
    """
    Fires a Discord webhook when the regime label changes.
    Includes a cooldown to prevent alert storms.
    """

    def __init__(
        self,
        webhook_url: str = "",
        *,
        cooldown_seconds: float = 300.0,
        enabled: bool = True,
    ):
        self.webhook_url = str(webhook_url or "")
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.enabled = bool(enabled) and bool(self.webhook_url)
        self._last_regime: str = ""
        self._last_alert_ts: float = 0.0

    def notify(self, new_regime: str, *, equity: float = 0.0, cycle: int = 0, extra: Optional[Dict[str, Any]] = None) -> bool:
        """Call each cycle with the current regime label. Returns True if alert sent."""
        new_regime = str(new_regime or "unknown")
        if not self.enabled or not self.webhook_url:
            self._last_regime = new_regime
            return False
        if new_regime == self._last_regime:
            return False
        now = time.time()
        if (now - self._last_alert_ts) < self.cooldown_seconds and self._last_regime:
            self._last_regime = new_regime
            return False
        old_regime = self._last_regime
        self._last_regime = new_regime
        self._last_alert_ts = now
        return self._send(old_regime=old_regime, new_regime=new_regime, equity=equity, cycle=cycle, extra=extra or {})

    def _send(self, *, old_regime: str, new_regime: str, equity: float, cycle: int, extra: Dict[str, Any]) -> bool:
        try:
            import urllib.request, json as _json
            eq_str = f"${equity:,.2f} AUD" if equity else "N/A"
            msg = (
                f"🔄 **Regime Change** | `{old_regime or 'none'}` → `{new_regime}` "
                f"| equity={eq_str} | cycle={cycle}"
            )
            if extra:
                extra_str = " ".join(f"{k}={v}" for k, v in list(extra.items())[:5])
                msg += f" | {extra_str}"
            payload = _json.dumps({"content": msg}).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                ok = 200 <= resp.status < 300
            if ok:
                logger.info("RegimeChangeAlerter: sent alert %s->%s", old_regime, new_regime)
            return ok
        except Exception as e:
            logger.debug("RegimeChangeAlerter send failed: %s", e)
            return False

    def reset(self) -> None:
        self._last_regime = ""
        self._last_alert_ts = 0.0
