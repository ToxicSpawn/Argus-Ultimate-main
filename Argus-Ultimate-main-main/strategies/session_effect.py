"""
Weekend/Session Effect Strategy — exploit systematic crypto time patterns.

Documented crypto time-of-day and day-of-week biases:
  - Friday 18:00-22:00 UTC: retail FOMO buying (slight bullish)
  - Sunday 00:00-06:00 UTC: low liquidity dump (bearish)
  - Monday 08:00-12:00 UTC: institutional buying (bullish)
  - US market open (13:30 UTC): volatility spike
  - Asia close (07:00 UTC): often reversal point

Also tracks:
  - Month-end rebalancing (last 2 days of month)
  - Quarterly options expiry (last Friday of quarter)
"""
from __future__ import annotations

import calendar
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Session definitions (UTC hours)
_SESSIONS = {
    "asia":    (0, 8),      # 00:00 - 08:00 UTC
    "london":  (8, 13),     # 08:00 - 13:00 UTC
    "ny":      (13, 21),    # 13:00 - 21:00 UTC
    "weekend": None,        # Saturday/Sunday (checked by day of week)
}

# Day-of-week patterns (0=Monday, 6=Sunday)
_DOW_BIAS = {
    0: ("bullish", 0.3, "Monday institutional buying"),
    1: ("neutral", 0.1, "Tuesday — no strong pattern"),
    2: ("neutral", 0.1, "Wednesday — mid-week neutral"),
    3: ("neutral", 0.1, "Thursday — pre-weekend positioning"),
    4: ("bullish", 0.2, "Friday — retail FOMO"),
    5: ("bearish", 0.15, "Saturday — low liquidity"),
    6: ("bearish", 0.25, "Sunday — low liquidity dump"),
}

# Intraday patterns (hour ranges → bias)
_INTRADAY_PATTERNS = [
    # (start_hour, end_hour, bias, strength, reasoning)
    (0, 6, "bearish", 0.2, "Sunday low-liquidity window (if Sunday)"),
    (7, 8, "neutral", 0.15, "Asia session close — potential reversal"),
    (8, 12, "bullish", 0.2, "Monday London/institutional buying (if Monday)"),
    (13, 14, "neutral", 0.3, "US market open — volatility spike"),
    (18, 22, "bullish", 0.15, "Friday retail FOMO window (if Friday)"),
]


class SessionEffectStrategy:
    """
    Exploit systematic crypto time-of-day and day-of-week patterns.

    Generates a session bias each cycle that can be used as a confidence
    multiplier or tiebreaker for other strategies.
    """

    def __init__(self, min_strength: float = 0.15) -> None:
        self._min_strength = min_strength

    # ------------------------------------------------------------------
    # Session detection
    # ------------------------------------------------------------------

    def get_current_session(self, now: Optional[datetime] = None) -> str:
        """Return current trading session name."""
        if now is None:
            now = datetime.now(timezone.utc)
        dow = now.weekday()
        hour = now.hour

        if dow >= 5:  # Saturday or Sunday
            return "weekend"

        for session_name, hours in _SESSIONS.items():
            if hours is None:
                continue
            start, end = hours
            if start <= hour < end:
                return session_name

        return "off_hours"

    def is_month_end(self, now: Optional[datetime] = None) -> bool:
        """Return True if in the last 2 days of the month."""
        if now is None:
            now = datetime.now(timezone.utc)
        _, last_day = calendar.monthrange(now.year, now.month)
        return now.day >= last_day - 1

    def is_quarterly_expiry(self, now: Optional[datetime] = None) -> bool:
        """
        Return True if today is the last Friday of a quarter month
        (March, June, September, December).
        """
        if now is None:
            now = datetime.now(timezone.utc)
        if now.month not in (3, 6, 9, 12):
            return False
        # Find last Friday of the month
        _, last_day = calendar.monthrange(now.year, now.month)
        last_date = datetime(now.year, now.month, last_day, tzinfo=timezone.utc)
        # Walk back to Friday (weekday 4)
        offset = (last_date.weekday() - 4) % 7
        last_friday = last_day - offset
        return now.day == last_friday

    # ------------------------------------------------------------------
    # Bias computation
    # ------------------------------------------------------------------

    def get_session_bias(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Return current session bias.

        Returns:
            {
                "bias": "bullish" | "bearish" | "neutral",
                "strength": 0.0 - 1.0,
                "session": session name,
                "reasoning": explanation string,
            }
        """
        if now is None:
            now = datetime.now(timezone.utc)

        session = self.get_current_session(now)
        dow = now.weekday()
        hour = now.hour

        # Start with day-of-week bias
        dow_bias, dow_strength, dow_reason = _DOW_BIAS.get(dow, ("neutral", 0.0, ""))

        # Layer intraday pattern on top
        intraday_bias = "neutral"
        intraday_strength = 0.0
        intraday_reason = ""

        for start_h, end_h, bias, strength, reason in _INTRADAY_PATTERNS:
            if start_h <= hour < end_h:
                # Check if pattern applies to this day
                if "Sunday" in reason and dow != 6:
                    continue
                if "Monday" in reason and dow != 0:
                    continue
                if "Friday" in reason and dow != 4:
                    continue
                intraday_bias = bias
                intraday_strength = strength
                intraday_reason = reason
                break

        # Combine: use the stronger signal
        if intraday_strength > dow_strength:
            bias = intraday_bias
            strength = intraday_strength
            reasoning = intraday_reason
        else:
            bias = dow_bias
            strength = dow_strength
            reasoning = dow_reason

        # Month-end boost
        if self.is_month_end(now):
            strength = min(1.0, strength + 0.1)
            reasoning += " + month-end rebalancing"

        # Quarterly expiry boost
        if self.is_quarterly_expiry(now):
            strength = min(1.0, strength + 0.15)
            reasoning += " + quarterly options expiry"

        # US market open volatility spike (13:30 - 14:00 UTC, weekdays)
        if dow < 5 and hour == 13 and now.minute >= 30:
            strength = min(1.0, strength + 0.2)
            reasoning += " + US market open volatility"

        return {
            "bias": bias,
            "strength": round(strength, 3),
            "session": session,
            "reasoning": reasoning.strip(),
            "hour_utc": hour,
            "day_of_week": dow,
            "is_month_end": self.is_month_end(now),
            "is_quarterly_expiry": self.is_quarterly_expiry(now),
        }

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signal(
        self,
        symbol: str,
        ohlcv: Any = None,
        regime: str = "NORMAL",
        now: Optional[datetime] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a signal when session bias aligns with other factors.

        Only generates a signal if session bias strength exceeds min_strength.
        Returns a dict compatible with TradingSignal, or None.
        """
        bias_info = self.get_session_bias(now)

        if bias_info["strength"] < self._min_strength:
            return None

        if bias_info["bias"] == "neutral":
            return None

        # Regime alignment check
        regime_upper = regime.upper()
        regime_aligns = False
        if bias_info["bias"] == "bullish" and regime_upper in (
            "TRENDING_UP", "TREND_UP", "BREAKOUT", "NORMAL",
        ):
            regime_aligns = True
        elif bias_info["bias"] == "bearish" and regime_upper in (
            "TRENDING_DOWN", "TREND_DOWN", "HIGH_VOL", "CRISIS",
        ):
            regime_aligns = True

        # Boost confidence if regime aligns
        confidence = bias_info["strength"]
        if regime_aligns:
            confidence = min(1.0, confidence + 0.15)

        action = "BUY" if bias_info["bias"] == "bullish" else "SELL"

        return {
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "strength": bias_info["strength"],
            "entry_price": 0.0,  # Caller should fill
            "reasoning": f"Session effect: {bias_info['reasoning']} ({bias_info['session']})",
            "strategy": "session_effect",
            "session": bias_info["session"],
            "regime_aligned": regime_aligns,
        }
